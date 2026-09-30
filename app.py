import streamlit as st
import ccxt
import pandas as pd
from datetime import datetime, time
import pytz

# Page configuration
st.set_page_config(page_title="EMA Breakout Crypto Screener", layout="wide")
st.title("📈 Crypto EMA Away & Breakout Screener")

# Initialize Binance exchange via CCXT
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

IST = pytz.timezone('Asia/Kolkata')

@st.cache_data(ttl=300)
def get_top_binance_symbols(limit):
    """Fetch top active USDT spot trading pairs sorted by 24h volume."""
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []
        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT') and ticker.get('quoteVolume') is not None:
                # Filter out active leveraged or index tokens
                if not any(x in symbol for x in ['UP/', 'DOWN/', 'BEAR/', 'BULL/']):
                    usdt_pairs.append({
                        'symbol': symbol,
                        'volume': ticker['quoteVolume']
                    })
        # Sort by 24h volume descending
        usdt_pairs = sorted(usdt_pairs, key=lambda x: x['volume'], reverse=True)
        return [p['symbol'] for p in usdt_pairs[:limit]]
    except Exception as e:
        st.error(f"Error fetching symbols from Binance: {e}")
        return []

def fetch_ohlcv_data(symbol, timeframe, limit=300):
    """Fetch OHLCV data from Binance and prepare dataframe."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime_ist'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(IST)
        # Drop the current unclosed live candle
        df = df.iloc[:-1].reset_index(drop=True)
        return df
    except Exception:
        return None

def get_ema_at_timestamp(df, ema_period, target_time):
    """Find the EMA value of the candle corresponding to or just before target_time."""
    if df is None or len(df) < ema_period:
        return None
    df = df.copy()
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    
    # Filter rows up to target_time
    valid_rows = df[df['datetime_ist'] <= target_time]
    if not valid_rows.empty:
        return valid_rows.iloc[-1]['ema']
    return None

def calculate_closest_ema_distance(high, low, ema_val):
    """Calculate whether High or Low is closer to EMA and return percentage string."""
    if ema_val is None or pd.isna(ema_val) or ema_val == 0:
        return "N/A"
    
    diff_high = abs(high - ema_val)
    diff_low = abs(low - ema_val)
    
    if diff_high <= diff_low:
        pct = ((high - ema_val) / ema_val) * 100
        sign = "+" if pct >= 0 else ""
        return f"High({sign}{pct:.2f}%)"
    else:
        pct = ((low - ema_val) / ema_val) * 100
        sign = "+" if pct >= 0 else ""
        return f"Low({sign}{pct:.2f}%)"

def check_away_and_breakout(df_scan, df_1h, df_4h, ema_period, breakout_type, date_filter_enabled, start_datetime, end_datetime):
    """Scan dataframe for EMA away + volume breakout and calculate HTF EMA distances."""
    if df_scan is None or len(df_scan) < ema_period + 5:
        return []

    # Calculate EMA for the scan timeframe
    df_scan['ema'] = df_scan['close'].ewm(span=ema_period, adjust=False).mean()
    
    matches = []

    # Loop through candles where EMA is available
    for i in range(ema_period, len(df_scan) - 1):
        row = df_scan.iloc[i]
        c_time = row['datetime_ist']

        # Apply IST Date & Time Filter if enabled
        if date_filter_enabled:
            if not (start_datetime <= c_time <= end_datetime):
                continue

        high, low, open_p, close_p, vol = row['high'], row['low'], row['open'], row['close'], row['volume']
        ema = row['ema']

        # Check if candle is completely away from EMA (Touch Nothing)
        is_away_above = (low > ema) and (high > ema) and (open_p > ema) and (close_p > ema)
        is_away_below = (high < ema) and (low < ema) and (open_p < ema) and (close_p < ema)

        if not (is_away_above or is_away_below):
            continue

        # Check the next up to 3 candles for breakout with volume
        for offset in range(1, 4):
            breakout_idx = i + offset
            if breakout_idx >= len(df_scan):
                break

            b_row = df_scan.iloc[breakout_idx]
            b_high, b_low, b_close, b_vol = b_row['high'], b_row['low'], b_row['close'], b_row['volume']

            # Volume condition: Breakout candle volume must be higher than Away candle volume
            if b_vol <= vol:
                continue

            breakout_detected = False
            direction = ""

            if is_away_above:
                # Bullish away -> Breakout above high
                if breakout_type == "High/Low Breakout" and b_high > high:
                    breakout_detected = True
                    direction = "Bullish High Break"
                elif breakout_type == "Close Breakout" and b_close > high:
                    breakout_detected = True
                    direction = "Bullish Close Break"

            elif is_away_below:
                # Bearish away -> Breakout below low
                if breakout_type == "High/Low Breakout" and b_low < low:
                    breakout_detected = True
                    direction = "Bearish Low Break"
                elif breakout_type == "Close Breakout" and b_close < low:
                    breakout_detected = True
                    direction = "Bearish Close Break"

            if breakout_detected:
                position_tag = "(U)" if is_away_above else "(D)"
                
                # Fetch 1h 51 EMA and 4h 51 EMA / 101 EMA at Away Candle Time
                ema_1h_51 = get_ema_at_timestamp(df_1h, 51, c_time)
                ema_4h_51 = get_ema_at_timestamp(df_4h, 51, c_time)
                ema_4h_101 = get_ema_at_timestamp(df_4h, 101, c_time)

                # Trend Circle Marking (4h 51 EMA vs 4h 101 EMA)
                trend_symbol = "🟢" if (ema_4h_51 is not None and ema_4h_101 is not None and ema_4h_51 > ema_4h_101) else "🔴"

                # Calculate distances
                dist_1h_51 = calculate_closest_ema_distance(high, low, ema_1h_51)
                dist_4h_51 = calculate_closest_ema_distance(high, low, ema_4h_51)
                dist_4h_101 = calculate_closest_ema_distance(high, low, ema_4h_101)

                matches.append({
                    'trend_symbol': trend_symbol,
                    'position_tag': position_tag,
                    'Away Candle Time (IST)': c_time.strftime('%Y-%m-%d %H:%M'),
                    'Away Price (High/Low)': f"{high} / {low}",
                    'Away Vol': f"{vol:,.2f}",
                    '1h 51 EMA Dist': dist_1h_51,
                    '4h 51 EMA Dist': dist_4h_51,
                    '4h 101 EMA Dist': dist_4h_101,
                    'Breakout Candle Time (IST)': b_row['datetime_ist'].strftime('%Y-%m-%d %H:%M'),
                    'Breakout Offset': f"{offset} candle(s) after",
                    'Breakout Close': b_close,
                    'Breakout Vol': f"{b_vol:,.2f}",
                    'Type': direction
                })
                # Once breakout found for this away candle, move to next away candle
                break

    return matches

# Sidebar UI Options
st.sidebar.header("⚙️ Screener Controls")

# 1. EMA Period Selection
ema_option = st.sidebar.selectbox("Select EMA Period:", [2, 3, 4, 5, 6], index=1)

# 2. Number of Top Binance Coins
top_coins_count = st.sidebar.selectbox("Select Top Coins Count (by Volume):", [50, 100, 200, 400, 500, 600, 700], index=1)

# 3. Timeframe Selection
tf_option = st.sidebar.selectbox("Select Timeframe:", ["15m", "30m", "1h", "2h", "4h", "1d"], index=2)

# 4. Breakout Type
breakout_option = st.sidebar.selectbox(
    "Select Breakout Type:", 
    ["Close Breakout", "High/Low Breakout"], 
    help="Close Breakout requires candle to close beyond away candle high/low. High/Low Breakout checks if wick/touch breaks high/low."
)

# 5. Date & Time Filter (IST)
st.sidebar.subheader("📅 Date & Time Filter (IST)")
enable_date = st.sidebar.checkbox("Enable Specific Date & Time Filter")

start_datetime, end_datetime = None, None

if enable_date:
    selected_date = st.sidebar.date_input("Select Date", datetime.now(IST))
    enable_time = st.sidebar.checkbox("Specify Time Range")
    
    if enable_time:
        start_time_val = st.sidebar.time_input("Start Time (IST)", time(0, 0))
        end_time_val = st.sidebar.time_input("End Time (IST)", time(23, 59))
        start_datetime = IST.localize(datetime.combine(selected_date, start_time_val))
        end_datetime = IST.localize(datetime.combine(selected_date, end_time_val))
    else:
        start_datetime = IST.localize(datetime.combine(selected_date, time(0, 0)))
        end_datetime = IST.localize(datetime.combine(selected_date, time(23, 59)))

# Run Screener Button
if st.button("🚀 Start Scanning"):
    st.info(f"Fetching Binance Top {top_coins_count} pairs and scanning...")
    symbols = get_top_binance_symbols(top_coins_count)

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, symbol in enumerate(symbols):
        status_text.text(f"Scanning ({idx + 1}/{len(symbols)}): {symbol}")
        
        # Fetch Scan Timeframe, 1h, and 4h data for distances/trends
        df_scan = fetch_ohlcv_data(symbol, tf_option)
        df_1h = fetch_ohlcv_data(symbol, "1h") if tf_option != "1h" else df_scan
        df_4h = fetch_ohlcv_data(symbol, "4h") if tf_option != "4h" else df_scan

        matches = check_away_and_breakout(
            df_scan, df_1h, df_4h, ema_option, breakout_option, enable_date, start_datetime, end_datetime
        )

        for match in matches:
            # Add Green/Red circle prefix and (U)/(D) suffix
            match['Symbol'] = f"{match['trend_symbol']} {symbol} {match['position_tag']}"
            results.append(match)

        progress_bar.progress((idx + 1) / len(symbols))

    status_text.text("Scanning Completed!")
    progress_bar.empty()

    if results:
        res_df = pd.DataFrame(results)
        cols = [
            'Symbol', 'Type', 'Away Candle Time (IST)', 'Away Price (High/Low)', 'Away Vol', 
            '1h 51 EMA Dist', '4h 51 EMA Dist', '4h 101 EMA Dist',
            'Breakout Candle Time (IST)', 'Breakout Offset', 'Breakout Close', 'Breakout Vol'
        ]
        res_df = res_df[cols]
        
        st.success(f"Found {len(res_df)} matching setup(s)!")
        st.dataframe(res_df, use_container_width=True)
    else:
        st.warning("Selected criteria par koi coin scan me nahi mila.")