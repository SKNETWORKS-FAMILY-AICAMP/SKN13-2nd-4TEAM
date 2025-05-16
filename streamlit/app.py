import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import joblib

st.set_page_config(page_title="📉 폐업 위험 예측 시스템", layout="wide")
plt.rcParams['font.family'] = 'Nanum Gothic'
plt.rcParams['axes.unicode_minus'] = False

@st.cache_resource
def load_code_name_dicts():
    return joblib.load("SKN13-2nd-4TEAM/산출물/best/dict_trans.joblib")

code_name_dicts = load_code_name_dicts()

@st.cache_data
def load_model_and_metadata(path):
    data = joblib.load(path)
    model = data.get('model', data)
    encoders = data.get('encoders', {})
    features = data.get('features', [])
    metrics = data.get('metrics', {})
    return model, encoders, features, metrics

def preprocess(df, encoder_dict):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        le = encoder_dict.get(col)
        if le:
            df[col] = le.transform(df[col])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def normalize_colname(col):
    return col.lower().replace('_','').replace(' ','')

def align_columns(df, expected_cols):
    df_cols = df.columns.tolist()
    expected_norm = {normalize_colname(c): c for c in expected_cols}
    col_map = {}
    for col in df_cols:
        norm_col = normalize_colname(col)
        if norm_col in expected_norm:
            col_map[col] = expected_norm[norm_col]
    df_renamed = df.rename(columns=col_map)
    return df_renamed

def add_korean_names(df):
    df = df.copy()
    if '상권구분코드' in df.columns:
        df['상권구분코드명'] = df['상권구분코드'].map(code_name_dicts['상권구분코드'])
    if '상권코드' in df.columns:
        df['상권코드명'] = df['상권코드'].map(code_name_dicts['상권코드'])
    if '서비스업종코드' in df.columns:
        df['서비스업종코드명'] = df['서비스업종코드'].map(code_name_dicts['서비스업종코드'])
    return df

MODEL_PATHS = {
    "CatBoost": r"SKN13-2nd-4TEAM/산출물/best/cb_model.joblib",
    "Random Forest": r"SKN13-2nd-4TEAM/산출물/best/rf_model.joblib"
}

DROP_COLS = [
    '상권_구분_코드', '상권_구분_코드_명', '상권_코드', '상권_코드_명',
    '서비스_업종_코드', '서비스_업종_코드_명',
    '주중_매출_건수', '주말_매출_건수',
    '월요일_매출_건수', '화요일_매출_건수', '수요일_매출_건수', '목요일_매출_건수', '금요일_매출_건수',
    '토요일_매출_건수', '일요일_매출_건수',
    '시간대_건수~06_매출_건수', '시간대_건수~11_매출_건수', '시간대_건수~14_매출_건수',
    '시간대_건수~17_매출_건수', '시간대_건수~21_매출_건수', '시간대_건수~24_매출_건수',
    '남성_매출_건수', '여성_매출_건수',
    '연령대_10_매출_건수', '연령대_20_매출_건수', '연령대_30_매출_건수', '연령대_40_매출_건수',
    '연령대_50_매출_건수', '연령대_60_이상_매출_건수'
]

uploaded_file = st.file_uploader("📂 예측할 CSV 파일", type=["csv"])

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    st.success("✅ CSV 업로드 완료. 모델 성능 비교 및 예측을 시작합니다.")
    test_df.drop(columns=DROP_COLS, errors='ignore', inplace=True)

    model_infos = []
    for name, path in MODEL_PATHS.items():
        model, encoders, features, metrics = load_model_and_metadata(path)
        model_infos.append({
            "name": metrics.get("model_name", name),
            "model": model,
            "encoders": encoders,
            "features": features,
            "f1": metrics.get("f1_score", 0),
            "recall": metrics.get("recall", 0)
        })

    st.subheader("📈 모델 성능 비교")
    st.table(pd.DataFrame({
        "모델 이름": [m["name"] for m in model_infos],
        "F1 Score": [f"{m['f1']:.4f}" for m in model_infos],
        "Recall": [f"{m['recall']:.4f}" for m in model_infos]
    }))

    best_model_info = sorted(model_infos, key=lambda x: (x['recall'], x['f1']), reverse=True)[0]
    st.success(f"✅ **{best_model_info['name']}** 모델이 선택되었습니다.")

    test_df_aligned = align_columns(test_df, best_model_info["features"])
    missing_cols = set(best_model_info["features"]) - set(test_df_aligned.columns)
    for col in missing_cols:
        test_df_aligned[col] = 0

    test_X = test_df_aligned[best_model_info["features"]].copy()
    test_X_scaled = preprocess(test_X, best_model_info["encoders"])

    with st.spinner("📊 예측 진행 중..."):
        model = best_model_info["model"]
        pred = model.predict(test_X_scaled)
        proba = model.predict_proba(test_X_scaled)[:, 1]
        test_df['폐업예측'] = pred
        test_df['폐업확률(%)'] = (proba * 100).round(2)
        test_df = add_korean_names(test_df)

    st.subheader("📋 예측 결과 미리보기")

    filter_cols = [
        ('상권구분코드명', '상권 구분 선택'),
        ('상권코드명', '상권 선택'),
        ('서비스업종코드명', '업종 선택')
    ]

    for col, label in filter_cols:
        if col in test_df.columns:
            options = sorted(test_df[col].dropna().unique())
            selected = st.selectbox(f"🔍 {label}", ['전체'] + options, key=col)
            if selected != '전체':
                test_df = test_df[test_df[col] == selected]

    st.dataframe(test_df.head(20))

    st.subheader("🔍 인사이트 컬럼 선택")
    column_options = {
        "상권 구분 코드명": "상권구분코드명",
        "상권 코드명": "상권코드명",
        "서비스 업종 코드명": "서비스업종코드명"
    }
    selected_cols = [col for label, col in column_options.items() if col in test_df.columns and st.checkbox(label, value=True)]

    st.subheader("💡 선택한 컬럼 기반 인사이트")
    with st.expander("📌 폐업 위험 높은 그룹 분석"):
        if selected_cols:
            grouped_df = test_df.groupby(selected_cols)['폐업확률(%)'].mean().reset_index()
            # 평균 폐업확률이 가장 높은 조합 찾기
            top_row = grouped_df.sort_values(by='폐업확률(%)', ascending=False).iloc[0]

            # 조합 문자열 생성
            combo_desc = ", ".join([f"{col} **{top_row[col]}**" for col in selected_cols])
            prob_value = top_row['폐업확률(%)']

            st.markdown(f"- {combo_desc}의 평균 폐업 확률은 **{prob_value:.2f}%**입니다.")
        else:
            st.info("인사이트를 보기 위해 하나 이상의 컬럼을 선택하세요.")


    display_cols = selected_cols + ['폐업예측', '폐업확률(%)']
    filtered_df = test_df[display_cols] if selected_cols else test_df[['폐업예측', '폐업확률(%)']]
    st.dataframe(filtered_df.head(20))

    if st.checkbox("🔎 Feature 중요도 보기"):
        feat_imp = pd.Series(model.feature_importances_, index=best_model_info["features"]).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=feat_imp.values[:15], y=feat_imp.index[:15], palette="viridis", ax=ax)
        ax.set_title("Feature Importance (Top 15)")
        st.pyplot(fig)

    if '위도' in test_df.columns and '경도' in test_df.columns:
        st.subheader("🗺️ 폐업 예측 지도")
        m = folium.Map(location=[test_df['위도'].mean(), test_df['경도'].mean()], zoom_start=13)
        for _, row in test_df.iterrows():
            color = "red" if row['폐업예측'] == 1 else "green"
            folium.CircleMarker(
                location=[row['위도'], row['경도']],
                radius=5,
                popup=f"폐업확률: {row['폐업확률(%)']:.1f}%",
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
        st_folium(m, width=700, height=500)
else:
    st.info("⬆️ 왼쪽에서 CSV 파일을 업로드해주세요.")
