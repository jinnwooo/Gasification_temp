import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from chardet.universaldetector import UniversalDetector
import os

# 함수: 파일의 인코딩을 감지
def detect_encoding(file_path):
    detector = UniversalDetector()
    with open(file_path, 'rb') as file:
        for line in file:
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    return detector.result['encoding']

def get_window_data(csv_file_or_df, window_size, scale_data=False):
    '''
    슬라이딩 윈도우를 사용하여 시계열 데이터를 전처리.

    :param csv_file_or_df: CSV 파일 경로 또는 DataFrame 객체
    :param window_size: 윈도우 크기 (스텝 수)
    :param scale_data: 데이터를 스케일링할지 여부 (기본값: False)
    :return: 입력 데이터(X), 타겟 데이터(y), 스케일러(scaler_X, scaler_y) (스케일링한 경우)
    '''
    # 데이터 로드
    if isinstance(csv_file_or_df, str):
        # 파일 경로인 경우
        encoding = detect_encoding(csv_file_or_df)
        df = pd.read_csv(csv_file_or_df, encoding=encoding)
    elif isinstance(csv_file_or_df, pd.DataFrame):
        # DataFrame 객체인 경우
        df = csv_file_or_df
    else:
        raise TypeError("csv_file_or_df는 파일 경로(str) 또는 DataFrame 객체여야 합니다.")

    # 불필요한 열 제거
    df = df.drop(['Hour', ' Min.', ' Sec.'], axis=1, errors='ignore')

    # 정적, 비정적 피처 및 타겟 변수 정의
    static_features = [
        'Actual feeding rate [g/min]',
        'Total N2 flow rate [g/min]',
        'O2 flow rate [g/min]',
        'CO2 flow rate [g/min]',
        'Steam flow rate [g/min]'
    ]
    non_static_features = [
        'FNC-1', 'FNC-2', 'FNC-3',
        'TI-2', 'TI-3', 'TI-4', 'TI-5',
        'PI-2', 'PI-3', 'PI-4', 'PI-5'
    ]
    target_columns = ['CO', 'CO2', 'CH4', 'H2']

    # 필요한 컬럼 확인
    required_columns = static_features + non_static_features + target_columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"데이터에 다음 컬럼이 없습니다: {missing_columns}")

    # 정적 피처 값 추출
    static_feature_values = df[static_features].iloc[0].values

    X_windows, y_windows = [], []
    for i in range(len(df) - window_size):
        # 현재 시점까지의 비정적 피처 데이터
        window_features = df[non_static_features].iloc[i:i+window_size+1].values.flatten()
        # 현재 시점 이전까지의 타겟 변수 데이터
        window_target_history = df[target_columns].iloc[i:i+window_size].values.flatten()
        # 현재 시점의 타겟 변수
        window_targets = df[target_columns].iloc[i+window_size].values

        # 정적 피처 + 현재 시점까지의 비정적 데이터 + 타겟 히스토리
        combined_window = np.hstack([static_feature_values, window_features, window_target_history])

        X_windows.append(combined_window)
        y_windows.append(window_targets)

    X, y = np.array(X_windows), np.array(y_windows)

    if scale_data:
        # 스케일링
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        return X_scaled, y_scaled, scaler_X, scaler_y
    else:
        return X, y


def get_feature_names(window_size):
    static_features = [
        'Actual feeding rate [g/min]',
        'Total N2 flow rate [g/min]',
        'O2 flow rate [g/min]',
        'CO2 flow rate [g/min]',
        'steam flow rate [g/min]'
    ]
    non_static_features = [
        'FNC-1', 'FNC-2', 'FNC-3',
        'TI-2', 'TI-3', 'TI-4', 'TI-5',
        'PI-2', 'PI-3', 'PI-4', 'PI-5'
    ]
    target_columns = ['CO', 'CO2', 'CH4', 'H2']

    # 정적 피처 이름
    feature_names = static_features.copy()

    # 비정적 피처 이름 생성
    for i in range(window_size+1):  # t까지 포함
        for feature in non_static_features:
            feature_names.append(f'{feature}_t-{window_size - i}')

    # 타겟 변수 히스토리 이름 생성
    for i in range(window_size):  # t-1까지
        for target in target_columns:
            feature_names.append(f'{target}_t-{window_size - i}')

    return feature_names


abnormal_0 = [
    "221028_mask o2steam gasification_with ac.csv",
    "20210517_MASK pellet CO2 가스화.csv",
    "20220105_mask_O2_steam_ gasification.csv"
]
abnormal_1 = [
    "20210504_mask pellet air gasificaion.csv",
    "20210528_steam 가스화 최종_(non A.C).csv",
    "20210621_MASK PELLET_CO2가스화_NON AC.csv"
]

def load_csv_with_issue_handling(csv_path):
    """
    - abnormal_0 -> return None
    - abnormal_1 -> 20분까지만
    - normal -> df 그대로
    """
    base = os.path.basename(csv_path)
    if base in abnormal_0:
        return None  # 완전 skip

    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None

    if base in abnormal_1:
        start_time = df[' Elapse(min.)'].iloc[0]
        df = df[df[' Elapse(min.)'] <= start_time + 20]
        if len(df) == 0:
            return None

    return df


def get_window_data_for_rnn(df, window_size, scale_data=False):
    """
    df -> (X, y, static_feat_vals) 생성
    예: df의 첫 행에서 5개 정적 피처를 뽑고,
        window_size 슬라이딩으로 (N, seq_len, dyn_feat_dim), (N, target_dim)
    """
    # 예시 전개(간단 버전)
    # 1) 정적 피처
    static_features = [
        'Actual feeding rate [g/min]',
        'Total N2 flow rate [g/min]',
        'O2 flow rate [g/min]',
        'CO2 flow rate [g/min]',
        'Steam flow rate [g/min]'
    ]
    static_vals = df[static_features].iloc[0].values  # shape (5,)

    # 2) 동적 피처
    non_static_cols = [
        'FNC-1', 'FNC-2', 'FNC-3',
        'TI-2', 'TI-3', 'TI-4', 'TI-5',
        'PI-2', 'PI-3', 'PI-4', 'PI-5'
    ]
    # input feature
    input_features = static_features + non_static_cols

    target_cols = ['CO', 'CO2', 'CH4', 'H2']

    # 3) 슬라이딩 윈도우로 X, y 생성
    X_list, y_list = [], []
    for i in range(len(df) - window_size):
        # 동적 피처: (window_size, 11)
        X_dynamic = df[input_features].iloc[i:i + window_size].values  # shape (window_size, 16)

        # 이전 타겟 피처: (window_size, 4)
        y_prev = df[target_cols].iloc[i:i + window_size].values  # shape (window_size, 4)

        # 동적 피처와 이전 타겟 피처를 합쳐서 입력 피처 생성: (window_size, 20)
        X_combined = np.hstack((X_dynamic, y_prev))  # shape (window_size, 20)

        # 다음 시점의 피처와 마스크 타겟
        x_next = df[input_features].iloc[i + window_size].values # shape (16,)
        y_mask = [-1, -1, -1, -1]

        x_next_with_mask = np.hstack((x_next, y_mask))  # shape (20,)
        X_combined = np.vstack((X_combined, x_next_with_mask))  # shape (window_size + 1, 20)
        # 타겟: 다음 시점의 타겟 값 (4,)
        y_next = df[target_cols].iloc[i + window_size].values  # shape (4,)

        X_list.append(X_combined)
        y_list.append(y_next)

    if len(X_list) == 0:
        return None, None

    X_array = np.array(X_list, dtype=np.float32)  # shape: (N, window_size, 15)
    y_array = np.array(y_list, dtype=np.float32)  # shape: (N, 4)

    if scale_data:
        # 스케일링
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_array = scaler_X.fit_transform(X_array.reshape(-1, X_array.shape[-1])).reshape(X_array.shape)
        y_array = scaler_y.fit_transform(y_array)
        return X_array, y_array, scaler_X, scaler_y

    return X_array, y_array
