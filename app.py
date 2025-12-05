import streamlit as st
import pandas as pd
import numpy as np
from nsepython import nse_optionchain_scrapper
from datetime import datetime, date, timedelta
import requests
from SmartApi import SmartConnect
import pyotp
from streamlit_autorefresh import st_autorefresh
import concurrent.futures
import time
import math
from scipy.stats import norm 

# --- PAGE CONFIG ---
st.set_page_config(page_title="Nifty Trap Master PRO (UI Fixed)", layout="wide", page_icon="🦁")

# --- UI STYLES ---
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #31333F; }
    /* CSS to ensure equal height for all cards */
    div[data-testid="column"] {
        display: flex;
        flex-direction: column;
    }
    .clean-card {
        background-color: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 10px;
        padding: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        color: #31333F; margin-bottom: 5px; 
        height: 100%; /* Forces full height */
        min-height: 110px; /* Minimum height to match Spot box */
        display: flex; flex-direction: column; justify-content: center; align-items: center;
    }
    .signal-box { padding: 20px; border-radius: 12px; text-align: center; margin: 10px 0; font-weight: bold; }
    .buy-signal { background-color: #E8F5E9; border: 2px solid #2E7D32; color: #1B5E20; }
    .sell-signal { background-color: #FFEBEE; border: 2px solid #C62828; color: #B71C1C; }
    .trap-signal { background-color: #FFF8E1; border: 2px solid #FF8F00; color: #BF360C; }
</style>
""", unsafe_allow_html=True)

# --- 1. MATHS LOGIC ---
def calculate_delta(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma == 0: return 0.5 if option_type == "CE" else -0.5
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return norm.cdf(d1) if option_type == 'CE' else norm.cdf(d1) - 1
    except Exception as e:
        return 0.5

def get_best_strike(df, option_type, target_delta):
    try:
        if option_type == 'CE':
            df['delta_diff'] = abs(df['CE Delta'] - target_delta)
        else:
            target_delta = -abs(target_delta)
            df['delta_diff'] = abs(df['PE Delta'] - target_delta)
        return df.sort_values('delta_diff').iloc[0]['Strike']
    except Exception:
        return 0

# --- 2. DATA FETCHING (FIXED EXCEPTION HANDLING) ---
@st.cache_data(ttl=3600)
def load_scrip_master():
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        data = requests.get(url).json()
        df = pd.DataFrame(data)
        
        # Filter Nifty Futures & Sort by Date
        df_fut = df[(df['exch_seg'] == 'NFO') & (df['name'] == 'NIFTY') & (df['instrumenttype'] == 'FUTIDX')].copy()
        df_fut['expiry_dt'] = pd.to_datetime(df_fut['expiry'], format='%d%b%Y', errors='coerce')
        df_fut = df_fut[df_fut['expiry_dt'] >= pd.Timestamp.today().normalize()]
        df_fut = df_fut.sort_values('expiry_dt')
        
        df_eq = df[df['exch_seg'] == 'NSE']
        return pd.concat([df_eq, df_fut])
    except Exception as e:
        st.error(f"🚨 CRITICAL ERROR: Failed to load Scrip Master. Check Internet. Details: {e}")
        return None

def fetch_nse_chain():
    try:
        payload = nse_optionchain_scrapper('NIFTY')
        if payload: return payload['records']['data'], payload['records']['expiryDates']
    except Exception as e:
        st.toast(f"⚠️ NSE Python Scraping Failed: {e}", icon="⚠️")
        pass
    return None, []

# --- 3. INDICATOR CALCULATIONS ---
def fetch_candles(api, token, exchange="NSE", interval="FIVE_MINUTE", days=1):
    try:
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        params = {
            "exchange": exchange, "symboltoken": token, "interval": interval,
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M")
        }
        data = api.getCandleData(params)
        if data['status'] and data['data']:
            df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=False) 
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['volume'] = df['volume'].astype(float)
            return df
        else:
            print(f"Candle Data Empty for {token}: {data.get('message', 'Unknown Error')}")
    except Exception as e:
        print(f"Candle Fetch Exception for {token}: {e}")
        pass
    return pd.DataFrame()

def calculate_indicators(api, spot_token, fut_token):
    rsi = 50; vwap = 0
    
    # RSI (Spot)
    try:
        df_spot = fetch_candles(api, spot_token, "NSE", "FIVE_MINUTE", 3)
        if not df_spot.empty:
            delta = df_spot['close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi = round(rsi.iloc[-1], 2)
    except Exception as e:
        print(f"RSI Calc Error: {e}")

    # VWAP (Future)
    try:
        df_fut = fetch_candles(api, fut_token, "NFO", "FIVE_MINUTE", 3) 
        if not df_fut.empty:
            last_candle_date = df_fut['timestamp'].iloc[-1].date()
            df_intraday = df_fut[df_fut['timestamp'].dt.date == last_candle_date].copy()
            if not df_intraday.empty:
                df_intraday['tp'] = (df_intraday['high'] + df_intraday['low'] + df_intraday['close']) / 3
                df_intraday['tp_vol'] = df_intraday['tp'] * df_intraday['volume']
                cum_tp_vol = df_intraday['tp_vol'].cumsum()
                cum_vol = df_intraday['volume'].cumsum()
                df_intraday['vwap'] = cum_tp_vol / cum_vol
                vwap = round(df_intraday['vwap'].iloc[-1], 2)
    except Exception as e:
        print(f"VWAP Calc Error: {e}")
        
    return rsi, vwap

# --- 4. STOCK LOGIC ---
def fetch_single_stock(api, symbol, token, name):
    try:
        quote = api.ltpData("NSE", symbol, token)
        if quote['status']:
            ltp = float(quote['data']['ltp'])
            open_price = float(quote['data'].get('open', ltp))
            close = float(quote['data'].get('close', ltp))
            pct_change = ((ltp - close) / close) * 100
            intraday_status = "Bearish" if ltp < open_price else "Bullish"
            weight = 2 if name in ['HDFC Bank', 'Reliance'] else 1
            score = (1 * weight) if intraday_status == "Bullish" else -(1 * weight)
            return name, pct_change, score, intraday_status
        else:
            print(f"LTP Failed for {symbol}")
    except Exception as e:
        print(f"Stock Fetch Exception {symbol}: {e}")
        pass
    return name, 0, 0, "Neutral"

def fetch_heavyweights(api, master_df):
    weights = {'HDFCBANK-EQ': 'HDFC Bank', 'RELIANCE-EQ': 'Reliance', 'ICICIBANK-EQ': 'ICICI Bank', 'INFY-EQ': 'Infosys', 'TCS-EQ': 'TCS'}
    results = {}; total_score = 0; details = {}
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for symbol, name in weights.items():
                try:
                    row = master_df[(master_df['symbol'] == symbol) & (master_df['exch_seg'] == 'NSE')]
                    if not row.empty:
                        token = row.iloc[0]['token']
                        futures.append(executor.submit(fetch_single_stock, api, symbol, token, name))
                except Exception as inner_e:
                    print(f"Master lookup failed for {symbol}: {inner_e}")

            for f in concurrent.futures.as_completed(futures):
                try:
                    name, pct, score, status = f.result()
                    results[name] = pct; details[name] = status; total_score += score
                except Exception as e:
                    print(f"Thread Result Error: {e}")
    except Exception as e:
        st.error(f"Heavyweights Thread Error: {e}")
        
    return results, total_score, details

# --- MAIN APP ---
st.title("🦁 Nifty Trap Master PRO (Full UI)")

if 'nse_data_cache' not in st.session_state: st.session_state['nse_data_cache'] = None
if 'last_nse_fetch_time' not in st.session_state: st.session_state['last_nse_fetch_time'] = 0

with st.sidebar:
    st.header("🔐 Login")
    api_key = st.text_input("API Key", type="password")
    client_id = st.text_input("Client ID")
    password = st.text_input("Password", type="password")
    totp = st.text_input("TOTP Secret", type="password")
    if st.button("Connect"):
        try:
            smartApi = SmartConnect(api_key=api_key)
            totp_obj = pyotp.TOTP(totp).now()
            data = smartApi.generateSession(client_id, password, totp_obj)
            if data['status']:
                st.session_state['angel_api'] = smartApi
                st.success("Connected!")
            else:
                st.error(f"Login Failed: {data.get('message', 'Invalid Credentials')}")
        except Exception as e: st.error(f"Connection Error: {e}")
    st_autorefresh(interval=10000, key="refresh")

with st.spinner("Analyzing Market..."):
    master_df = load_scrip_master()

if 'angel_api' in st.session_state and master_df is not None:
    api = st.session_state['angel_api']
    
    # A. FETCH REAL-TIME COMPONENTS
    comp_data, comp_score, comp_details = fetch_heavyweights(api, master_df)
    
    # B. FETCH NIFTY DATA & CALCULATE INDICATORS
    spot_price = 0; vix_price = 0; fut_ltp = 0
    rsi_val = 50; vwap_val = 0
    
    try:
        spot_pkt = api.ltpData("NSE", "Nifty 50", "99926000")
        vix_pkt = api.ltpData("NSE", "INDIA VIX", "26009")
        if spot_pkt['status']: spot_price = float(spot_pkt['data']['ltp'])
        if vix_pkt['status']: vix_price = float(vix_pkt['data']['ltp'])
        
        # Get NEAREST Future
        nifty_fut = master_df[(master_df['exch_seg'] == 'NFO') & (master_df['name'] == 'NIFTY') & (master_df['instrumenttype'] == 'FUTIDX')]
        if not nifty_fut.empty:
            cur_fut = nifty_fut.sort_values('expiry_dt').iloc[0]
            fut_token = cur_fut['token']
            
            # Get Future LTP & API VWAP
            fut_pkt = api.ltpData("NFO", cur_fut['symbol'], fut_token)
            api_vwap = 0
            if fut_pkt['status']: 
                fut_ltp = float(fut_pkt['data']['ltp'])
                api_vwap = float(fut_pkt['data'].get('averagePrice', 0))

            # Manual VWAP
            rsi_val, vwap_val = calculate_indicators(api, "99926000", fut_token)
            if vwap_val == 0: vwap_val = api_vwap if api_vwap > 0 else fut_ltp
        else:
            st.warning("Future Token Not Found in Master")

    except Exception as e: st.error(f"Main Data Loop Error: {e}")

    # C. OPTION CHAIN
    curr_time = time.time()
    if curr_time - st.session_state['last_nse_fetch_time'] > 30 or st.session_state['nse_data_cache'] is None:
        data, exps = fetch_nse_chain()
        if data:
            st.session_state['nse_data_cache'] = data
            st.session_state['nse_expiry_cache'] = exps
            st.session_state['last_nse_fetch_time'] = curr_time

    raw_chain = st.session_state['nse_data_cache']
    expiry_list = st.session_state.get('nse_expiry_cache', [])

    if raw_chain and spot_price > 0:
        sel_exp = st.selectbox("📅 Expiry", expiry_list, index=0)
        
        try:
            exp_date = datetime.strptime(sel_exp, "%d-%b-%Y").date()
            T = max((exp_date - date.today()).days / 365.0, 1e-5)
            
            chain_data = []
            tot_ce_oi = 0; tot_pe_oi = 0 

            for item in raw_chain:
                if item['expiryDate'] == sel_exp:
                    ce = item.get('CE', {}); pe = item.get('PE', {})
                    strike = item['strikePrice']
                    
                    ce_delta = calculate_delta(spot_price, strike, T, 0.1, ce.get('impliedVolatility',0)/100, 'CE')
                    pe_delta = calculate_delta(spot_price, strike, T, 0.1, pe.get('impliedVolatility',0)/100, 'PE')
                    
                    c_oi = ce.get('openInterest', 0); p_oi = pe.get('openInterest', 0)
                    tot_ce_oi += c_oi; tot_pe_oi += p_oi
                    
                    chain_data.append({
                        'Strike': strike, 'CE OI': c_oi, 'CE Delta': round(ce_delta, 2),
                        'PE OI': p_oi, 'PE Delta': round(pe_delta, 2)
                    })

            df = pd.DataFrame(chain_data).sort_values('Strike')
            atm = round(spot_price / 50) * 50
            df_view = df[(df['Strike'] >= atm - 800) & (df['Strike'] <= atm + 800)]
            pcr_val = tot_pe_oi / tot_ce_oi if tot_ce_oi > 0 else 0

            # --- SCORING & ACTION (UPDATED WITH IMAGE LOGIC) ---
            bull_score = 0; bear_score = 0; reasons = []
            
            # 1. Stocks Logic (Flexible)
            if comp_score >= 1: bull_score += 2; reasons.append("Stocks Mild +")
            if comp_score > 3: bull_score += 2; reasons.append("Stocks Strong +")
            
            if comp_score <= -1: bear_score += 2; reasons.append("Stocks Mild -")
            if comp_score < -3: bear_score += 2; reasons.append("Stocks Strong -")
            
            # 2. VWAP Logic
            if fut_ltp > vwap_val: bull_score += 3; reasons.append("Fut > VWAP")
            elif fut_ltp < vwap_val: bear_score += 3; reasons.append("Fut < VWAP")
            
            # 3. RSI Logic
            if rsi_val > 55: bull_score += 1
            elif rsi_val < 45: bear_score += 1
            
            # 4. PCR LOGIC (STRICTLY AS PER YOUR IMAGE)
            pcr_msg = "Neutral"; pcr_color = "#31333F"
            
            if pcr_val > 1.5:
                bear_score += 2; reasons.append("PCR > 1.5 (Reversal Down)")
                pcr_msg = "Bearish Reversal"; pcr_color = "red"
            elif 1.1 <= pcr_val <= 1.5:
                bull_score += 2; reasons.append("PCR Bullish")
                pcr_msg = "Bullish Trend"; pcr_color = "green"
            elif 0.95 < pcr_val < 1.1:
                reasons.append("PCR Sideways")
                pcr_msg = "Sideways"; pcr_color = "#E65100" # Orange
            elif 0.65 <= pcr_val <= 0.95:
                bear_score += 2; reasons.append("PCR Bearish")
                pcr_msg = "Bearish Trend"; pcr_color = "red"
            elif pcr_val < 0.65:
                bull_score += 2; reasons.append("PCR < 0.65 (Reversal Up)")
                pcr_msg = "Bullish Reversal"; pcr_color = "green"

            # 5. Support/Resistance (Range Increased to 60)
            sup_strike = df_view.loc[df_view['PE OI'].idxmax(), 'Strike']
            res_strike = df_view.loc[df_view['CE OI'].idxmax(), 'Strike']
            if abs(spot_price - sup_strike) < 60: bull_score += 2; reasons.append("Near Support")
            if abs(spot_price - res_strike) < 60: bear_score += 2; reasons.append("Near Resistance")

            safe_ce = get_best_strike(df_view, 'CE', 0.50); fast_ce = get_best_strike(df_view, 'CE', 0.70)
            safe_pe = get_best_strike(df_view, 'PE', 0.50); fast_pe = get_best_strike(df_view, 'PE', 0.70)
            
            action_msg = "WAIT & WATCH"; action_color = "#FFF8E1"; action_txt_color = "#BF360C"; border="#FF8F00"
            rec_html = ""; sl_html = ""; tgt_html = ""
            is_trap = False
            
            # Trap Checks
            if (comp_score > 0 and fut_ltp < vwap_val) or (comp_score < 0 and fut_ltp > vwap_val):
                is_trap = True; action_msg = "⚠️ TRAP: Data Mismatch"
            if not is_trap:
                if rsi_val > 75: is_trap = True; action_msg = "⚠️ TRAP: Overbought"
                elif rsi_val < 25: is_trap = True; action_msg = "⚠️ TRAP: Oversold"

            if not is_trap:
                # Entry Threshold set to 6
                if bull_score >= 6 and fut_ltp > vwap_val:
                    action_msg = "🟢 STRONG BUY (CE)"; action_color = "#E8F5E9"; action_txt_color = "#1B5E20"; border="#2E7D32"
                    rec_html = f"Safe: {safe_ce} CE | Fast: {fast_ce} CE"
                    sl_val = sup_strike - 10
                    sl_html = f"SL: {sl_val}"
                    t1_val = int(spot_price + 50)
                    tgt_html = f"T1: {t1_val} | T2: {res_strike}"
                    
                elif bear_score >= 6 and fut_ltp < vwap_val:
                    action_msg = "🔴 STRONG BUY (PE)"; action_color = "#FFEBEE"; action_txt_color = "#B71C1C"; border="#C62828"
                    rec_html = f"Safe: {safe_pe} PE | Fast: {fast_pe} PE"
                    sl_val = res_strike + 10
                    sl_html = f"SL: {sl_val}"
                    t1_val = int(spot_price - 50)
                    tgt_html = f"T1: {t1_val} | T2: {sup_strike}"

            # --- DISPLAY ---
            if "STRONG BUY" in action_msg:
                st.markdown(f"""
                <div class='signal-box { 'buy-signal' if 'CE' in action_msg else 'sell-signal' }'>
                    <h1>{action_msg}</h1>
                    <p>{' + '.join(reasons)}</p>
                    <hr>
                    <div style='display:flex; justify-content:space-between; margin-bottom:5px;'>
                        <span>🎯 {rec_html}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between'>
                        <span style='color:blue'>🚀 {tgt_html}</span>
                        <span style='color:red'>🛡️ {sl_html}</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='signal-box trap-signal'><h3>{action_msg}</h3><p>{' + '.join(reasons)}</p></div>""", unsafe_allow_html=True)

            # 6-COLUMN DASHBOARD (FIXED HEIGHTS & PCR MESSAGE)
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            
            # SPOT
            m1.markdown(f"<div class='clean-card'><small>NIFTY SPOT</small><br><b>{spot_price}</b><br><small>&nbsp;</small></div>", unsafe_allow_html=True)
            
            # FUT / VWAP (STACKED)
            vwap_col = "green" if fut_ltp > vwap_val else "red"
            m2.markdown(f"<div class='clean-card'><small>FUT / VWAP</small><br><span>{int(fut_ltp)}</span><br><b style='color:{vwap_col}'>{vwap_val}</b></div>", unsafe_allow_html=True)
            
            # RSI
            rsi_col = "red" if rsi_val > 70 else ("green" if rsi_val < 30 else "#31333F")
            m3.markdown(f"<div class='clean-card'><small>RSI (5m)</small><br><b style='color:{rsi_col}'>{rsi_val}</b><br><small>&nbsp;</small></div>", unsafe_allow_html=True)

            # PCR (UPDATED WITH MESSAGE)
            m4.markdown(f"<div class='clean-card'><small>PCR</small><br><b style='color:{pcr_color}'>{pcr_val:.2f}</b><br><small style='color:{pcr_color}'>{pcr_msg}</small></div>", unsafe_allow_html=True)
            
            # CONFIDENCE
            conf_bg = "#E8F5E9" if bull_score > bear_score else "#FFEBEE"
            conf_txt = "#1B5E20" if bull_score > bear_score else "#B71C1C"
            m5.markdown(f"<div class='clean-card' style='background-color:{conf_bg}; color:{conf_txt}'><small>CONFIDENCE</small><br><b>{max(bull_score, bear_score)}/10</b><br><small>&nbsp;</small></div>", unsafe_allow_html=True)
            
            # OI LEVELS
            m6.markdown(f"<div class='clean-card'><small>OI LEVELS</small><br><span style='color:green'>S: {sup_strike}</span><br><span style='color:red'>R: {res_strike}</span></div>", unsafe_allow_html=True)

            st.write("")
            c1, c2, c3, c4, c5 = st.columns(5)
            cols = [c1, c2, c3, c4, c5]
            for i, (name, val) in enumerate(comp_details.items()):
                color = "green" if val == "Bullish" else "red"
                cols[i].markdown(f"<div class='clean-card' style='border-top: 3px solid {color}'><b>{name}</b><br><span style='color:{color}'>{val}</span></div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Display/Logic Error: {e}")

else: st.info("Please Login First")