import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from joblib import Parallel, delayed
import time

# 시계열 관련 라이브러리
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.datasets import load_airline, load_shampoo_sales, load_lynx
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import STLTransformer
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA
from statsmodels.tsa.stattools import acf, pacf

# streamlit 제목 설정
st.set_page_config(
    page_title="C221061 전태환 프로젝트 1",
    page_icon="📈"
)
st.header("📈 시계열 예측 대시보드 (C221061 전태환)")

# --- 1. 유틸리티 함수 ---

def evaluate_forecast(y_true, y_pred, y_train=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-10
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    TS = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(np.diff(y_true))) + eps)
    
    if y_train is not None:
        naive_error = np.mean(np.abs(np.diff(np.array(y_train))))
        mase = mae / (naive_error + eps)
    else:
        mase = np.nan
    return {"MAE": mae, "MSE": mse, "MAPE(%)": mape, "MASE": mase, "TS": TS}

def fit_and_predict_single_model(name, model, y_train, fh):
    try:
        model.fit(y_train)
        y_pred = model.predict(fh)
        return name, y_pred, None
    except Exception as e:
        return name, None, str(e)

def safe_to_timestamp(index):
    """sktime의 PeriodIndex를 Plotly 시각화를 위해 DatetimeIndex로 변환"""
    if hasattr(index, 'to_timestamp'):
        return index.to_timestamp()
    return index

# --- 2. 핵심 실행 함수 (캐싱 적용) ---
@st.cache_data(show_spinner=False)
def run_fast_forecasting(y_series, test_size, sp, selected_models):
    # 데이터 분할
    y_train, y_test = temporal_train_test_split(y_series, test_size=test_size)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    actual_sp = max(2,sp)

    all_configs = {
        "Naive": NaiveForecaster(strategy="last"),
        "SMA": NaiveForecaster(strategy="mean"),
        "Exp Smoothing": ExponentialSmoothing(trend=None, seasonal=None),
        "Holt-Winters (Add)": ExponentialSmoothing(trend="add", seasonal="add", sp=actual_sp),
        "STL Forecaster": TransformedTargetForecaster(steps=[
            ("stl", STLTransformer(sp=actual_sp)),
            ("forecast", NaiveForecaster(strategy="drift"))
        ]),
        "AutoARIMA": AutoARIMA(
            sp=actual_sp, 
            seasonal=(actual_sp > 1), # sp가 1이면 seasonal은 False여야 함
            stepwise=True,
            suppress_warnings=True, 
            start_p=0, start_q=0,
            max_p=1, max_q=1, # 범위를 조금만 넓혀주되 제한함
            start_P=1, start_Q=0,
            max_P=1, max_Q=0, # 범위를 조금만 넓혀주되 제한함
            trace=True # Streamlit 환경에서는 False 권장
        )
    }

    model_configs = {name: all_configs[name] for name in selected_models if name in all_configs}
    
    # 결과를 담을 딕셔너리
    preds = {}
    
    # 병렬 처리 대신 루프를 돌려 에러 메시지 상세 확인 (디버깅용)
    for name, model in model_configs.items():
        try:
            model.fit(y_train)
            y_pred = model.predict(fh)
            preds[name] = y_pred
        except Exception as e:
            st.error(f"모델 {name} 실행 중 에러 발생: {e}") # 화면에 에러 출력
            
    return y_train, y_test, preds

def preprocess_uploaded_csv(uploaded_file):
    """
    업로드된 CSV 파일을 읽어 시계열 데이터(y)와 기본 계절성 주기(sp)를 반환함
    """
    try:
        # 1. 데이터 로드
        df = pd.read_csv(uploaded_file)
        
        # 2. 사이드바에서 컬럼 선택 (함수 내부에 UI를 포함)
        st.sidebar.subheader("📍 컬럼 지정")
        date_col = st.sidebar.selectbox("날짜(또는 연도) 열", df.columns)
        val_col = st.sidebar.selectbox("예측 대상(Value) 열", df.columns)
        
        # 3. 날짜 변환 및 인덱스 설정
        # 숫자로 된 연도(1950 등) 처리를 위해 astype(str) 적용
        df[date_col] = pd.to_datetime(df[date_col].astype(str), errors='coerce')
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        
        # 4. 빈도(Frequency) 추론 및 매핑
        inferred_freq = pd.infer_freq(df.index)
        
        if inferred_freq:
            # Pandas 빈도 기호를 단순 기호로 표준화
            if any(x in inferred_freq for x in ['Y', 'A', 'AS', 'YS']):
                freq = 'Y'
            elif any(x in inferred_freq for x in ['Q', 'QS', 'BQ']):
                freq = 'Q'
            elif any(x in inferred_freq for x in ['M', 'MS']):
                freq = 'M'
            else:
                freq = 'D'
        else:
            # 추론 실패 시 사용자에게 직접 물어봄
            freq_opt = {"연단위 (Y)": "Y", "분기단위 (Q)": "Q", "월단위 (M)": "M", "일단위 (D)": "D"}
            selected = st.sidebar.selectbox("데이터 주기 수동 선택", list(freq_opt.keys()))
            freq = freq_opt[selected]
        
        # 5. PeriodIndex 변환 및 중복 제거 (sktime 호환성 핵심)
        df.index = df.index.to_period(freq)
        y_input = df[val_col].groupby(level=0).mean()
        
        # 6. 계절성 주기(sp) 결정
        sp_map = {'Y': 1, 'Q': 4, 'M': 12, 'D': 7}
        sp_default = sp_map.get(freq, 1)
        
        return y_input, sp_default

    except Exception as e:
        st.error(f"데이터 처리 중 오류 발생: {e}")
        return None, None
    
@st.cache_data(show_spinner=False)
def load_builtin_data(source):
    if source == "Airline (월별)":
        return load_airline(), 12
    elif source == "Shampoo (월별)":
        return load_shampoo_sales(), 12
    elif source == "Lynx (연별)":
        return load_lynx(), 10
    return None, 12

def get_sidebar_data_config():
    """사이드바 UI를 렌더링하고 설정된 데이터를 로드함"""
    st.sidebar.header("📁 데이터 설정")
    source = st.sidebar.selectbox("데이터 소스 선택", ["Airline (월별)", "Shampoo (월별)", "Lynx (연별)", "CSV 직접 업로드"])
    
    y = None
    sp = 12
    
    if source == "CSV 직접 업로드":
        uploaded_file = st.sidebar.file_uploader("CSV 업로드", type=["csv"])
        if uploaded_file:
            # 앞서 논의한 CSV 전처리 로직 실행
            y, sp = preprocess_uploaded_csv(uploaded_file) 
    else:
        # 내장 데이터 로드 로지 (load_airline 등)
        y, sp = load_builtin_data(source)
        
    return y, sp, source

def detect_hampel_filter(series, window_size=5, n_sigmas=3):
    """
    Hampel Filter를 사용하여 이상치 인덱스를 반환합니다.
    """
    new_series = series.copy()
    k = 1.4826  # 정규분포 가정을 위한 상수
    
    # Rolling window로 중앙값과 MAD 계산
    rolling_median = series.rolling(window=2*window_size+1, center=True).median()
    rolling_mad = series.rolling(window=2*window_size+1, center=True).apply(lambda x: np.median(np.abs(x - np.median(x))))
    
    threshold = n_sigmas * k * rolling_mad
    difference = np.abs(series - rolling_median)
    
    # 차이가 임계값보다 큰 지점의 인덱스 탐색
    outlier_indices = np.where(difference > threshold)[0]
    return outlier_indices

# --- 사이드바 설정 추가 ---

st.set_page_config(layout="wide")
y_input, sp_default, data_name = get_sidebar_data_config()

with st.sidebar.expander("🔍 이상치 자동 탐지 설정 (Hampel)"):
    hampel_window = st.slider("Hampel Window Size", 1, 20, 5)
    hampel_sigma = st.slider("Hampel Sigma (민감도)", 1.0, 5.0, 3.0, step=0.5)
    show_hampel = st.checkbox("자동 탐지 지점 표시", value=True)


# 1. 세션 상태 초기화 (수정된 데이터를 유지하기 위함)
if 'modified_y' not in st.session_state:
    st.session_state.modified_y = y_input.copy()

if y_input is not None:
    st.sidebar.header("⚙️ 하이퍼파라미터")
    
    model_options = ["Naive", "SMA", "Exp Smoothing", "Holt-Winters (Add)", "STL Forecaster", "AutoARIMA"]
    selected_models = st.sidebar.multiselect(
        "분석할 모델 선택", 
        options=model_options, 
        default=["Naive", "SMA", "Exp Smoothing", "Holt-Winters (Add)", "STL Forecaster"]
    )
    
    max_test = max(1, len(y_input)//2)
    default_test = min(12, max_test)
    test_size_input = st.sidebar.slider("예측 기간 (Test Size)", 1, max_test, default_test)
    seasonal_period = st.sidebar.number_input("계절성 주기 (sp)", value=sp_default, min_value=1)
    

   
        # --- 사이드바 설정 영역 ---
    with st.sidebar.expander("📈 차분 설정"):
        use_diff = st.checkbox("1차 차분 적용 (Stationarity)", value=False)
    
    if st.sidebar.button("♻️ 데이터 초기화 버튼", use_container_width=True):
        st.session_state.modified_y = y_input.copy()
        st.toast("데이터가 원본으로 복구되었습니다.") # 상단에 작은 알림
        st.rerun()
    
    predict_btn = st.sidebar.button("🚀 예측 시작하기",use_container_width=True)
    # --- 2. 메인 화면 영역 ---
    if predict_btn:
        if not selected_models:
            st.warning("⚠️ 최소 하나 이상의 모델을 선택해주세요.")
        else:
            start_time = time.time()
            
            # 1. 원본 데이터 준비
            raw_y = st.session_state.modified_y.copy()
            
            # 2. 차분 적용 여부에 따른 모델 입력 데이터 결정
            if use_diff:
                # 1차 차분 수행
                model_input_y = raw_y.diff().dropna()
                # 복원을 위해 '차분 전' 마지막 Train 데이터의 실제값 저장
                # (Test Size만큼 뺀 위치의 값)
                split_idx = len(raw_y) - test_size_input
                last_train_value = raw_y.iloc[split_idx - 1] 
            else:
                model_input_y = raw_y

            with st.spinner(f'{data_name} 모델 연산 중...'):
                # 모델 연산 (차분된 데이터 혹은 원본 입력)
                y_train, y_test, predictions = run_fast_forecasting(
                    model_input_y, test_size_input, seasonal_period, selected_models
                )
                
                # --- [핵심] 차분 데이터 복원 로직 ---
                if use_diff:
                    # 시각화를 위해 y_train과 y_test는 원본 스케일로 교체
                    y_train_raw, y_test_raw = temporal_train_test_split(raw_y, test_size=test_size_input)
                    y_train = y_train_raw
                    y_test = y_test_raw

                    # 모델별 예측치(predictions) 복원
                    for name in predictions:
                        # 차분 예측값들의 누적합을 구하고, 마지막 학습 시점의 원본값을 더함
                        # 복원 공식: 원본(t) = 원본(t-1) + 차분(t)
                        restored_pred = predictions[name].cumsum() + last_train_value
                        predictions[name] = restored_pred
                
            elapsed_time = time.time() - start_time
            st.sidebar.success(f"✅ 연산 완료! ({elapsed_time:.2f}초)")

            # 📊 그래프 시각화 (기존 코드와 동일하지만 데이터는 복원됨)
            st.subheader(f"📊 {data_name} 분석 결과" + (" (차분 복원 완료)" if use_diff else ""))
            fig = go.Figure()
            
            # Train (회색), Actual (남색)
            fig.add_trace(go.Scatter(x=safe_to_timestamp(y_train.index), y=y_train, name='Train', line=dict(color='#BDC3C7')))
            fig.add_trace(go.Scatter(x=safe_to_timestamp(y_test.index), y=y_test, name='Actual', line=dict(color='#2C3E50', width=2.5)))

            # 예측치 (주황색 계열 테마 적용)
            orange_shades = ['#FF8C00', '#FF4500', '#E67E22', '#D35400']
            for i, (name, y_pred) in enumerate(predictions.items()):
                color = orange_shades[i % len(orange_shades)]
                fig.add_trace(go.Scatter(
                    x=safe_to_timestamp(y_pred.index), 
                    y=y_pred, 
                    name=f"- {name}", 
                    line=dict(color=color, dash='dash', width=2)
                ))

            fig.update_layout(xaxis_type='date', height=550, template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # 🎯 성능 평가는 원본 스케일(y_test_raw) 기준으로 수행됨
            st.divider()
            st.subheader("🎯 모델 성능 평가")
            res_metrics = {n: evaluate_forecast(y_test, p, y_train) for n, p in predictions.items()}
            st.dataframe(pd.DataFrame(res_metrics).T.style.highlight_min(axis=0, color='#D5F5E3'), width="stretch")

            # res_metrics 시각화
            metric_names = list(next(iter(res_metrics.values())).keys())
            for metric in metric_names:
                fig_metric = go.Figure()
                for name in predictions:
                    fig_metric.add_trace(go.Bar(
                        x=[name], 
                        y=[res_metrics[name][metric]], 
                        name=name, 
                        marker_color=orange_shades[model_options.index(name) % len(orange_shades)]
                    ))
                fig_metric.update_layout(title=f"{metric} 비교", height=300, showlegend=False)
                st.plotly_chart(fig_metric, use_container_width=True)

        good_button = st.button("👍", use_container_width=True)
    else:
        # --- 데이터 미리보기 및 이상치 정제 영역 ---
        st.info("💡 빨간색 원은 Hampel Filter가 탐지한 이상치 후보입니다. 점을 클릭하여 수정한 뒤 예측을 시작하세요.")
        
        current_y = st.session_state.modified_y
        y_train_pre, y_test_pre = temporal_train_test_split(current_y, test_size=test_size_input)
        
        # 1. Hampel Filter 탐지
        hampel_outliers = detect_hampel_filter(current_y, window_size=hampel_window, n_sigmas=hampel_sigma)
        
        fig = go.Figure()

        # [그래프] 학습 데이터 (초록)
        fig.add_trace(go.Scatter(
            x=safe_to_timestamp(y_train_pre.index), y=y_train_pre, 
            name='Train Set', mode='lines+markers', marker=dict(size=5), line=dict(color="#27AE60")
        ))
        
        # [그래프] 테스트 데이터 (주황)
        fig.add_trace(go.Scatter(
            x=safe_to_timestamp(y_test_pre.index), y=y_test_pre, 
            name='Test Set', mode='lines+markers', marker=dict(size=5), line=dict(color="#E67E22", width=3)
        ))

        # [그래프] Hampel 이상치 강조 (빨간색 원)
        if show_hampel and len(hampel_outliers) > 0:
            fig.add_trace(go.Scatter(
                x=safe_to_timestamp(current_y.index[hampel_outliers]),
                y=current_y.iloc[hampel_outliers],
                mode='markers',
                name='Hampel Detection',
                marker=dict(
                    color='rgba(0,0,0,0)', 
                    size=15, 
                    line=dict(color='red', width=2)
                ),
                hoverinfo='skip'
            ))

        # 2. Plotly 차트 출력 및 선택 이벤트 캡처
        selected_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

        # 3. [데이터 수정 로직] 선택된 점이 있을 때만 UI 표시
        if selected_data and "selection" in selected_data and len(selected_data["selection"]["points"]) > 0:
            indices = [p["point_index"] for p in selected_data["selection"]["points"]]
            
            st.markdown(f"### 🛠️ 선택된 {len(indices)}개 지점 처리")
            
            c1, c2 = st.columns([2, 1])
            with c1:
                method = st.radio(
                    "보간 방법 선택",
                    ["직전 값 복사 (ffill)", "앞뒤 선형 보간 (Linear)", "1주기 전 값으로 대체", "구간 평균으로 대체"],
                    horizontal=True
                )
                
                if method == "구간 평균으로 대체":
                    adj_window = st.slider("평균 범위 (Window)", 1, 20, 3)

            with c2:
                if st.button("✨ 선택 지점 적용하기", use_container_width=True):
                    new_y = current_y.copy()
                    for idx in indices:
                        if method == "직전 값 복사 (ffill)":
                            if idx > 0: new_y.iloc[idx] = new_y.iloc[idx-1]
                        elif method == "앞뒤 선형 보간 (Linear)":
                            if 0 < idx < len(new_y)-1:
                                new_y.iloc[idx] = (new_y.iloc[idx-1] + new_y.iloc[idx+1]) / 2
                        elif method == "1주기 전 값으로 대체":
                            if idx >= seasonal_period:
                                new_y.iloc[idx] = new_y.iloc[idx - seasonal_period]
                        elif method == "구간 평균으로 대체":
                            start_idx = max(0, idx - adj_window)
                            end_idx = min(len(new_y), idx + adj_window + 1)
                            surrounding = pd.concat([new_y.iloc[start_idx:idx], new_y.iloc[idx+1:end_idx]])
                            if not surrounding.empty:
                                new_y.iloc[idx] = surrounding.mean()
                    
                    st.session_state.modified_y = new_y
                    st.success("데이터가 성공적으로 수정되었습니다!")
                    st.rerun()

        # --- [추가] 4. 자기 상관 함수 (ACF) 분석 영역 ---
        st.divider()
        col_acf1, col_acf2 = st.columns([2, 1])
        with col_acf1:
            st.subheader("🔍 상관 함수 분석 (ACF & PACF)")
            
            y_values = current_y.values
            n_obs = len(y_values)
            n_lags = min(n_obs // 4, 20) # 데이터의 1/4 혹은 최대 20까지만
            
            # ACF & PACF 계산
            acf_data = acf(y_values, nlags=n_lags)
            pacf_data = pacf(y_values, nlags=n_lags)
            
            # 두 그래프를 탭으로 나누거나 위아래로 배치
            tab1, tab2 = st.tabs(["ACF (주기성/추세)", "PACF (직접 영향)"])
            
            with tab1:
                fig_acf = go.Figure(go.Bar(x=list(range(len(acf_data))), y=acf_data, marker_color='#34495E'))
                fig_acf.add_hline(y=1.96/np.sqrt(len(y_values)), line_dash="dash", line_color="red")
                fig_acf.update_layout(height=300, margin=dict(t=10, b=10))
                st.plotly_chart(fig_acf, use_container_width=True)
                
            with tab2:
                fig_pacf = go.Figure(go.Bar(x=list(range(len(pacf_data))), y=pacf_data, marker_color='#E74C3C'))
                fig_pacf.add_hline(y=1.96/np.sqrt(len(y_values)), line_dash="dash", line_color="red")
                fig_pacf.update_layout(height=300, margin=dict(t=10, b=10))
                st.plotly_chart(fig_pacf, use_container_width=True)


        st.divider()

        st.subheader("🔄 차분 결과 미리보기")

        current_y = st.session_state.modified_y
        diff_y = current_y.diff().dropna()

        tab_raw, tab_diff = st.tabs(["원본 데이터", "1차 차분 데이터"])

        with tab_raw:
            st.line_chart(current_y)
            st.caption("현재 정제된 데이터의 모습입니다.")

        with tab_diff:
            st.line_chart(diff_y)
            st.caption("차분을 적용하면 평균이 0으로 수렴하며 추세가 제거됩니다. (정상성 확보)")

else:
    st.info("데이터를 로드하면 분석 대시보드가 활성화됩니다.")

