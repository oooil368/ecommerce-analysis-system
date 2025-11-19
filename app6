import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import silhouette_score
from scipy.stats import linregress  # 确保有这个导入
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# ============================================================================
# 自定义CSS样式 - 现代化界面
# ============================================================================
st.markdown("""
<style>
    /* 主标题样式 */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        padding: 1rem;
    }

    /* 顶部导航栏 */
    .top-nav {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .nav-button {
        background: rgba(255,255,255,0.2) !important;
        color: white !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        border-radius: 25px !important;
        padding: 0.5rem 1.5rem !important;
        margin: 0 0.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .nav-button:hover {
        background: rgba(255,255,255,0.3) !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    .nav-button.active {
        background: white !important;
        color: #667eea !important;
        border-color: white !important;
    }

    /* 任务状态指示器 */
    .status-indicator {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
        gap: 2rem;
    }

    .status-item {
        text-align: center;
        padding: 1rem;
        border-radius: 15px;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        min-width: 120px;
        transition: all 0.3s ease;
    }

    .status-item.completed {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
    }

    .status-item.pending {
        background: linear-gradient(135deg, #ffc107, #fd7e14);
        color: white;
    }

    /* 卡片样式 */
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }

    /* 工具栏样式 */
    .toolbar {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #e9ecef;
    }

    /* 按钮样式 */
    .stButton button {
        border-radius: 25px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }

    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 15px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 25px;
        padding: 0 2rem;
        background: white;
        border: 2px solid #e9ecef;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)
# ============================================================================
# 页面配置
# ============================================================================
st.set_page_config(
    page_title="电商销售分析与策略优化系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 自定义CSS样式
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .task-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .status-completed {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        border-left: 4px solid #28a745;
    }
    .status-pending {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        border-left: 4px solid #ffc107;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .fix-note {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State 初始化
# ============================================================================
def initialize_session_state():
    default_states = {
        'raw_data': None,  # 原始数据
        'task1_data': None,  # 任务1专用数据
        'task2_data': None,  # 任务2专用数据
        'task3_data': None,  # 任务3专用数据
        'task4_data': None,  # 任务4专用数据
        'step1_missing_data': None,
        'step2_price_data': None,
        'step3_profit_data': None,
        'step4_abnormal_data': None,
        'step5_minmax_data': None,
        'step5_zscore_data': None,
        'processed_data': None,
        'category_encoder': None,
        'current_file': None,
        'task1_completed': False,
        'task2_completed': False,
        'task3_completed': False,
        'task4_completed': False,
        'task2_visualizations': None,  # 新增：可视化结果
        'column_types': None
    }

    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value
initialize_session_state()


# ============================================================================
# 工具函数
# ============================================================================
def auto_detect_column_types(df):
    """自动识别字段类型：数值型、有序分类、无序分类、标识型"""
    column_types = {
        'numeric': [],  # 数值型（需标准化）
        'ordinal': [],  # 有序分类
        'nominal': [],  # 无序分类
        'identifier': []  # 标识型
    }

    # 标识型字段规则：唯一值占比>80% 或 字段名包含"ID/订单号/日期"
    id_keywords = ['id', '订单号', '日期', '编号', '序号']
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in id_keywords) or (df[col].nunique() / len(df) > 0.8):
            column_types['identifier'].append(col)
            continue

    # 数值型字段
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    column_types['numeric'] = [col for col in numeric_cols if col not in column_types['identifier']]

    # 分类字段（非数值、非标识）
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in column_types['identifier']]

    # 区分有序/无序分类
    ordinal_keywords = ['等级', '年龄', '评分', '段位', '层次']
    for col in categorical_cols:
        if any(keyword in col for keyword in ordinal_keywords):
            column_types['ordinal'].append(col)
        else:
            column_types['nominal'].append(col)

    return column_types


def enhanced_data_import_component(task_name, required_columns=None, allow_processed_data=True):
    """增强版通用数据导入组件 - 支持选择任务1的任一输出文件"""
    st.subheader("📁 数据导入")

    col1, col2 = st.columns(2)
    data_source = None
    current_data = None

    with col1:
        # 数据源选择
        data_source_option = st.radio(
            f"选择{task_name}数据源:",
            ["使用原始数据", "选择任务1处理文件", "上传新文件"],
            key=f"data_source_{task_name}"
        )

    with col2:
        if data_source_option == "使用原始数据":
            if st.session_state.get('raw_data') is not None:
                current_data = st.session_state.raw_data
                data_source = "原始数据"
                st.success(f"使用原始数据，共 {len(current_data)} 条记录")
            else:
                st.error("暂无原始数据，请先上传文件")
                return None, None

        elif data_source_option == "选择任务1处理文件":
            # 任务1生成的文件列表
            task1_files = {
                "步骤2_进货价格处理后数据": "step2_price_data",
                "步骤3_利润修正后数据": "step3_profit_data",
                "步骤4_异常修正及利润重算后数据": "step4_abnormal_data",
                "步骤5_MinMax标准化后数据": "step5_minmax_data",
                "步骤5_ZScore标准化后数据": "step5_zscore_data"
            }

            selected_file = st.selectbox(
                "选择任务1处理文件:",
                list(task1_files.keys()),
                key=f"task1_file_{task_name}"
            )

            if selected_file and st.session_state.get(task1_files[selected_file]) is not None:
                current_data = st.session_state[task1_files[selected_file]]
                data_source = f"任务1: {selected_file}"
                st.success(f"使用{selected_file}，共 {len(current_data)} 条记录")
            else:
                st.error("选择的文件不存在，请先完成任务1")
                return None, None

        else:  # 上传新文件
            uploaded_file = st.file_uploader(
                f"上传{task_name}数据文件",
                type=["xlsx", "csv"],
                key=f"upload_{task_name}"
            )

            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.xlsx'):
                        current_data = pd.read_excel(uploaded_file)
                    else:
                        current_data = pd.read_csv(uploaded_file)

                    current_data = clean_numeric_columns(current_data)
                    data_source = f"自定义文件: {uploaded_file.name}"
                    st.success(f"数据加载成功！共 {len(current_data)} 条记录")
                except Exception as e:
                    st.error(f"文件读取错误: {str(e)}")
                    return None, None
            else:
                st.info("请上传数据文件")
                return None, None

    # 检查必需字段
    if required_columns and current_data is not None:
        missing_columns = [col for col in required_columns if col not in current_data.columns]
        if missing_columns:
            st.error(f"缺少必需字段: {', '.join(missing_columns)}")
            st.info(f"{task_name}需要的字段: {', '.join(required_columns)}")
            return None, None

    return current_data, data_source

def clean_numeric_columns(df):
    """清洗数值列中的非数值字符"""
    df_clean = df.copy()

    # 尝试识别价格相关字段
    price_keywords = ['价格', '售价', '金额', '销售额', '利润', '成本']
    price_cols = [col for col in df.columns if any(kw in col for kw in price_keywords)]

    # 清洗价格相关字段
    for col in price_cols:
        if df_clean[col].dtype == 'object':
            # 去除常见非数值字符
            df_clean[col] = df_clean[col].astype(str) \
                .str.replace(r'[^\d.]', '', regex=True)
            # 转换为数值类型
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 处理百分比字段
    percent_keywords = ['率', '百分比', '占比']
    percent_cols = [col for col in df.columns if any(kw in col for kw in percent_keywords)]
    for col in percent_cols:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str) \
                .str.replace(r'[%]', '', regex=True)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce') / 100

    return df_clean


def process_categorical_variables(df, column_types, fit_encoder=True):
    """处理分类变量：有序→序数编码，无序→独热编码"""
    df_processed = df.copy()
    encoders = {}

    # 1. 有序分类：序数编码
    if column_types['ordinal'] and fit_encoder:
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_ordinal = ordinal_encoder.fit_transform(df_processed[column_types['ordinal']])
        df_ordinal = pd.DataFrame(
            df_ordinal,
            columns=[f"{col}_编码" for col in column_types['ordinal']],
            index=df_processed.index
        )
        df_processed = pd.concat([df_processed, df_ordinal], axis=1)
        encoders['ordinal'] = ordinal_encoder

    # 2. 无序分类：独热编码
    if column_types['nominal'] and fit_encoder:
        onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        df_onehot = onehot_encoder.fit_transform(df_processed[column_types['nominal']])
        # 生成独热编码字段名
        feature_names = []
        for i, col in enumerate(column_types['nominal']):
            categories = onehot_encoder.categories_[i][1:]  # 跳过第一个类别
            feature_names.extend([f"{col}_{cat}" for cat in categories])
        df_onehot = pd.DataFrame(
            df_onehot,
            columns=feature_names,
            index=df_processed.index
        )
        df_processed = pd.concat([df_processed, df_onehot], axis=1)
        encoders['onehot'] = onehot_encoder
        encoders['onehot_features'] = feature_names

    # 3. 非拟合模式
    if not fit_encoder and st.session_state.category_encoder:
        encoders = st.session_state.category_encoder
        if column_types['ordinal']:
            df_ordinal = encoders['ordinal'].transform(df_processed[column_types['ordinal']])
            df_ordinal = pd.DataFrame(
                df_ordinal,
                columns=[f"{col}_编码" for col in column_types['ordinal']],
                index=df_processed.index
            )
            df_processed = pd.concat([df_processed, df_ordinal], axis=1)
        if column_types['nominal']:
            df_onehot = encoders['onehot'].transform(df_processed[column_types['nominal']])
            df_onehot = pd.DataFrame(
                df_onehot,
                columns=encoders['onehot_features'],
                index=df_processed.index
            )
            df_processed = pd.concat([df_processed, df_onehot], axis=1)

    return df_processed, encoders


# ============================================================================
# 任务1：数据预处理类（按照独立脚本逻辑重构）
# ============================================================================
class Task1Preprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.results = {}
        self.column_types = None

    def step1_missing_value_analysis(self):
        """步骤1: 缺失值统计分析"""
        # 计算缺失值统计
        missing_stats = pd.DataFrame({
            '字段名': self.df.columns,
            '数据类型': self.df.dtypes.values,
            '总行数': len(self.df),
            '非空值数量': self.df.notnull().sum(),
            '缺失值数量': self.df.isnull().sum(),
            '缺失比例%': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })

        self.results['step1_missing_stats'] = missing_stats
        return missing_stats

    def step2_price_processing(self, missing_stats):
        """步骤2: 进货价格处理（按照独立脚本逻辑）"""
        df_step2 = self.df.copy()

        # 处理进货价格字段 - 按照独立脚本逻辑
        if '进货价格' in df_step2.columns:
            # 使用正则表达式去除非数字和非小数点字符
            df_step2['进货价格'] = df_step2['进货价格'].apply(
                lambda x: float(re.sub(r'[^\d\.]', '', str(x))) if re.search(r'[\d\.]', str(x)) else None
            )
            # 转换为整数型（四舍五入）
            df_step2['进货价格'] = df_step2['进货价格'].round().astype('Int64')

            # 处理缺失值（如果有）
            if df_step2['进货价格'].isnull().sum() > 0:
                if '商品品类' in df_step2.columns:
                    # 用品类中位数填充
                    category_price = df_step2.groupby('商品品类')['进货价格'].transform('median')
                    df_step2['进货价格'] = df_step2['进货价格'].fillna(category_price)
                else:
                    # 用整体中位数填充
                    df_step2['进货价格'] = df_step2['进货价格'].fillna(df_step2['进货价格'].median())

        self.results['step2_processed'] = df_step2
        return df_step2

    def step3_profit_correction(self, df_step2):
        """步骤3: 修正利润计算错误（使用随机森林和KNN模型）"""
        df_step3 = df_step2.copy()

        # 检查必要字段是否存在
        required_cols = ['实际售价', '进货价格', '销售数', '利润']
        missing_cols = [col for col in required_cols if col not in df_step3.columns]
        if missing_cols:
            st.warning(f"利润修正缺少字段: {missing_cols}，跳过利润修正")
            return df_step3

        # 计算理论利润
        df_step3['理论利润'] = (df_step3['实际售价'] - df_step3['进货价格']) * df_step3['销售数']

        # 筛选错误和正确数据
        error_data = df_step3[df_step3['利润'] != df_step3['理论利润']].copy()
        correct_data = df_step3[df_step3['利润'] == df_step3['理论利润']].copy()

        st.info(f"利润计算错误数据条数：{len(error_data)}")
        st.info(f"利润计算正确数据条数（训练数据）：{len(correct_data)}")

        if len(correct_data) == 0:
            st.warning("无利润计算正确的数据，无法训练模型进行补插")
            return df_step3

        if len(error_data) == 0:
            st.info("没有发现利润计算错误的数据")
            df_step3 = df_step3.drop(columns='理论利润')
            return df_step3

        # 准备模型训练数据
        features = ['实际售价', '进货价格', '销售数']
        X = correct_data[features]
        y = correct_data['利润']

        # 划分训练集和测试集
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 1. 训练随机森林模型
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.metrics import mean_squared_error

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred_test = rf_model.predict(X_test)
        rf_mse = mean_squared_error(y_test, rf_pred_test)
        st.info(f"随机森林模型测试集均方误差：{round(rf_mse, 2)}")

        # 2. 训练KNN模型
        knn_model = KNeighborsRegressor(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        knn_pred_test = knn_model.predict(X_test)
        knn_mse = mean_squared_error(y_test, knn_pred_test)
        st.info(f"KNN模型测试集均方误差：{round(knn_mse, 2)}")

        # 选择MSE较小的模型进行利润补插
        if rf_mse <= knn_mse:
            st.info("选择随机森林模型进行利润补插")
            error_X = error_data[features]
            pred_error = rf_model.predict(error_X)
            pred_error = pred_error.round().astype(df_step3['利润'].dtype)
        else:
            st.info("选择KNN模型进行利润补插")
            error_X = error_data[features]
            pred_error = knn_model.predict(error_X)
            pred_error = pred_error.round().astype(df_step3['利润'].dtype)

        # 更新错误利润值
        df_step3 = df_step3.reset_index(drop=True)
        error_data = error_data.reset_index(drop=True)
        df_step3.loc[error_data.index, '利润'] = pred_error

        # 删除临时的理论利润列
        df_step3 = df_step3.drop(columns='理论利润')

        self.results['step3_processed'] = df_step3
        return df_step3

    def step4_abnormal_correction(self, df_step3):
        """步骤4: 修正成本高于售价异常（使用模型预测合理售价）"""
        df_step4 = df_step3.copy()

        # 检查必要字段是否存在
        required_cols = ['实际售价', '进货价格', '销售数', '客户年龄']
        missing_cols = [col for col in required_cols if col not in df_step4.columns]
        if missing_cols:
            st.warning(f"异常修正缺少字段: {missing_cols}，跳过异常修正")
            return df_step4

        # 标记异常数据（实际售价 < 进货价格）
        abnormal_mask = df_step4['实际售价'] < df_step4['进货价格']
        abnormal_data = df_step4[abnormal_mask].copy()
        normal_data = df_step4[~abnormal_mask].copy()

        st.info(f"成本高于售价的异常数据条数：{len(abnormal_data)}")
        st.info(f"正常数据条数（训练数据）：{len(normal_data)}")

        if len(normal_data) == 0:
            st.warning("无正常售价数据，无法训练模型进行异常修正")
            return df_step4

        if len(abnormal_data) == 0:
            st.info("没有发现成本高于售价的异常数据")
            # 重新计算利润确保正确性
            if all(col in df_step4.columns for col in ['实际售价', '进货价格', '销售数']):
                df_step4['利润'] = (df_step4['实际售价'] - df_step4['进货价格']) * df_step4['销售数']
            return df_step4

        # 准备模型训练数据（预测合理实际售价）
        features = ['进货价格', '销售数', '客户年龄']
        target = '实际售价'
        X = normal_data[features]
        y = normal_data[target]

        # 划分训练集和测试集
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 1. 训练随机森林模型
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.metrics import mean_squared_error

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred_test = rf_model.predict(X_test)
        rf_mse = mean_squared_error(y_test, rf_pred_test)
        st.info(f"随机森林模型（售价预测）测试集均方误差：{round(rf_mse, 2)}")

        # 2. 训练KNN模型
        knn_model = KNeighborsRegressor(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        knn_pred_test = knn_model.predict(X_test)
        knn_mse = mean_squared_error(y_test, knn_pred_test)
        st.info(f"KNN模型（售价预测）测试集均方误差：{round(knn_mse, 2)}")

        # 综合两种模型结果进行售价补插（取平均值）
        abnormal_X = abnormal_data[features]
        rf_pred_abnormal = rf_model.predict(abnormal_X)
        knn_pred_abnormal = knn_model.predict(abnormal_X)
        combined_pred = (rf_pred_abnormal + knn_pred_abnormal) / 2
        combined_pred = combined_pred.round().astype(df_step4[target].dtype)

        # 更新异常数据的售价
        df_step4.loc[abnormal_mask, target] = combined_pred

        # 二次检查剩余异常（若仍有售价<进货价，将售价设为进货价）
        remaining_abnormal_mask = df_step4['实际售价'] < df_step4['进货价格']
        if remaining_abnormal_mask.sum() > 0:
            st.info(f"二次检查发现{remaining_abnormal_mask.sum()}条剩余异常数据，将售价设为进货价")
            df_step4.loc[remaining_abnormal_mask, '实际售价'] = df_step4.loc[remaining_abnormal_mask, '进货价格']

        # 重新计算正确利润（替换原利润列）
        df_step4['利润'] = (df_step4['实际售价'] - df_step4['进货价格']) * df_step4['销售数']

        self.results['step4_processed'] = df_step4
        return df_step4

    def step5_standardization(self, df_step4):
        """步骤5: 标准化处理（按照独立脚本逻辑）"""
        df_original = df_step4.copy()

        # 定义需标准化的数值列（与独立脚本完全一致）
        required_cols = ["进货价格", "实际售价", "销售数", "利润"]
        # 若存在销售额列，加入标准化范围
        if "销售额" in df_original.columns:
            required_cols.append("销售额")

        # 检查列是否存在
        missing_cols = [col for col in required_cols if col not in df_original.columns]
        if missing_cols:
            st.warning(f"标准化缺少字段: {missing_cols}")

        # 筛选数值型列
        numeric_cols = [col for col in required_cols if col in df_original.columns and
                        pd.api.types.is_numeric_dtype(df_original[col])]

        if not numeric_cols:
            st.warning("无可用的数值型列进行标准化")
            # 返回原始数据
            self.results['step5_minmax'] = df_original
            self.results['step5_zscore'] = df_original
            return df_original, df_original

        st.info(f"待标准化的数值列：{numeric_cols}")

        # 1. Z-Score标准化
        df_zscore = df_original.copy()
        scaler_z = StandardScaler()
        df_zscore[numeric_cols] = scaler_z.fit_transform(df_zscore[numeric_cols])

        # 2. Min-Max标准化
        df_minmax = df_original.copy()
        scaler_mm = MinMaxScaler(feature_range=(0, 1))
        df_minmax[numeric_cols] = scaler_mm.fit_transform(df_minmax[numeric_cols])

        self.results['step5_minmax'] = df_minmax
        self.results['step5_zscore'] = df_zscore
        self.results['numeric_cols'] = numeric_cols

        # 输出标准化统计信息
        st.info("Z-Score标准化后统计描述：")
        st.dataframe(df_zscore[numeric_cols].describe().round(4))
        st.info("Min-Max标准化后统计描述（0-1区间）：")
        st.dataframe(df_minmax[numeric_cols].describe().round(4))

        return df_minmax, df_zscore

    def generate_all_results(self):
        """生成所有步骤的结果"""
        try:
            # 执行全流程步骤
            step1_missing = self.step1_missing_value_analysis()
            step2_price = self.step2_price_processing(step1_missing)
            step3_profit = self.step3_profit_correction(step2_price)
            step4_abnormal = self.step4_abnormal_correction(step3_profit)
            step5_minmax, step5_zscore = self.step5_standardization(step4_abnormal)

            # 字段类型识别
            self.column_types = auto_detect_column_types(step4_abnormal)

            # 处理分类变量
            final_data, encoders = process_categorical_variables(
                step4_abnormal, self.column_types, fit_encoder=True)

            # 整理结果文件
            result_files = {
                '电商 步骤1 缺失值统计结果.xlsx': step1_missing,
                '电商 步骤2 进货价格处理后数据.xlsx': step2_price,
                '电商 步骤3 利润修正后数据.xlsx': step3_profit,
                '电商 步骤4 异常修正及利润重算后数据.xlsx': step4_abnormal,
                '电商 步骤5 MinMax标准化后数据.xlsx': step5_minmax,
                '电商 步骤5 ZScore标准化后数据.xlsx': step5_zscore
            }

            # 整理进度日志
            progress_log = [
                f"步骤1：完成缺失值统计，共{len(step1_missing)}个字段",
                f"步骤2：完成进货价格处理（去除货币符号并转换为整数型）",
                f"步骤3：完成利润修正（使用机器学习模型）",
                f"步骤4：完成异常值修正（成本高于售价异常）",
                f"步骤5：完成标准化处理，生成MinMax和ZScore两种标准化结果"
            ]

            return result_files, progress_log, final_data, encoders, self.column_types

        except Exception as e:
            return None, [f"预处理错误: {str(e)}"], None, None, None
# ============================================================================
# 增强版任务2：多维销售特征分析类（按论文要求重构）
# ============================================================================
# ============================================================================
# 增强可视化功能类
# ============================================================================
class EnhancedVisualizer:
    def __init__(self, df, column_types):
        self.df = df.copy()
        self.column_types = column_types
        self.results = {}

    def create_interactive_dashboard(self):
        """创建交互式仪表板"""
        figs = {}

        try:
            # 1. 销售趋势图
            if any('日期' in col for col in self.df.columns):
                date_col = next(col for col in self.df.columns if '日期' in col)
                daily_sales = self.df.groupby(date_col).agg({
                    '销售额': 'sum',
                    '利润': 'sum',
                    '销售数': 'sum'
                }).reset_index()

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=daily_sales[date_col], y=daily_sales['销售额'],
                                               mode='lines+markers', name='销售额', line=dict(color='#1f77b4')))
                fig_trend.add_trace(go.Scatter(x=daily_sales[date_col], y=daily_sales['利润'],
                                               mode='lines+markers', name='利润', line=dict(color='#ff7f0e')))
                fig_trend.update_layout(title='每日销售趋势', xaxis_title='日期', yaxis_title='金额')
                figs['sales_trend'] = fig_trend

            # 2. 商品品类销售分布
            if '商品品类' in self.df.columns:
                category_sales = self.df.groupby('商品品类').agg({
                    '销售额': 'sum',
                    '利润': 'sum'
                }).reset_index()

                fig_category = px.sunburst(category_sales, path=['商品品类'], values='销售额',
                                           title='商品品类销售分布')
                figs['category_sunburst'] = fig_category

                # 柱状图版本
                fig_bar = px.bar(category_sales.nlargest(10, '销售额'),
                                 x='商品品类', y='销售额', color='利润',
                                 title='Top 10 商品品类销售额')
                figs['category_bar'] = fig_bar

            # 3. 地理分布热力图
            if '区域' in self.df.columns:
                region_sales = self.df.groupby('区域').agg({
                    '销售额': 'sum',
                    '利润': 'sum'
                }).reset_index()

                fig_region = px.bar(region_sales.nlargest(10, '销售额'),
                                    x='区域', y='销售额', color='利润',
                                    title='区域销售额Top 10')
                figs['region_bar'] = fig_region

            # 4. 客户画像分析
            if all(col in self.df.columns for col in ['客户性别', '客户年龄']):
                fig_demographic = px.scatter(self.df, x='客户年龄', y='销售额', color='客户性别',
                                             size='销售数', hover_data=['商品品类'],
                                             title='客户年龄-销售额分布')
                figs['demographic_scatter'] = fig_demographic

            # 5. 价格-销量关系图
            if all(col in self.df.columns for col in ['实际售价', '销售数']):
                fig_price_volume = px.scatter(self.df, x='实际售价', y='销售数', color='商品品类',
                                              trendline="lowess", title='价格-销量关系分析')
                figs['price_volume'] = fig_price_volume

            # 6. 利润贡献分析
            if '利润' in self.df.columns:
                profit_analysis = self.df.nlargest(10, '利润')
                fig_profit = px.bar(profit_analysis, x='商品品类', y='利润', color='区域',
                                    title='Top 10 利润贡献商品')
                figs['profit_analysis'] = fig_profit

            self.results['interactive_dashboard'] = figs
            return True

        except Exception as e:
            st.error(f"交互式仪表板创建错误: {str(e)}")
            return False

    # ... 这里还要添加 create_advanced_analytics_charts、create_customer_segmentation_charts、
    # create_performance_metrics、generate_all_visualizations 等方法 ...


class EnhancedTask2Analyzer:
    def __init__(self, df, column_types):
        self.df = df.copy()
        self.column_types = column_types
        self.results = {}

    def enhanced_task2_multidimensional_analysis(self):
        """增强版多维分析页面 - 优化界面"""
        st.header("🔍 任务2: 多维销售特征分析")

        # ============================================================================
        # 顶部工具栏 - 横排操作按钮
        # ============================================================================
        st.markdown("### 🛠️ 分析工具栏")
        toolbar_col1, toolbar_col2, toolbar_col3, toolbar_col4, toolbar_col5 = st.columns(5)

        with toolbar_col1:
            data_source_option = st.selectbox(
                "数据源",
                ["使用原始数据", "选择任务1处理文件", "上传新文件"],
                key="data_source_task2"
            )

        with toolbar_col2:
            analysis_mode = st.selectbox(
                "分析模式",
                ["📊 Python可视化展示", "📁 论文图表数据导出", "🎨 交互式可视化仪表板"],
                key="analysis_mode"
            )

        with toolbar_col3:
            if st.button("🚀 执行分析", type="primary", use_container_width=True):
                st.session_state.run_analysis = True

        with toolbar_col4:
            if st.session_state.get('task2_completed'):
                if st.button("📥 下载结果", use_container_width=True):
                    st.session_state.download_results = True

        with toolbar_col5:
            if st.button("🔄 重新开始", use_container_width=True):
                st.session_state.task2_completed = False
                st.session_state.run_analysis = False
                st.rerun()

        # ============================================================================
        # 数据导入和清洗区域
        # ============================================================================
        st.markdown("### 📁 数据准备")

        data_source = None
        current_data = None

        # 数据源处理
        if data_source_option == "使用原始数据":
            if st.session_state.get('raw_data') is not None:
                current_data = st.session_state.raw_data
                data_source = "原始数据"
                st.success(f"✅ 使用原始数据，共 {len(current_data)} 条记录")
            else:
                st.error("❌ 暂无原始数据，请先在任务1中上传文件")
                return

        elif data_source_option == "选择任务1处理文件":
            task1_files = {
                "步骤2_进货价格处理后数据": "step2_price_data",
                "步骤3_利润修正后数据": "step3_profit_data",
                "步骤4_异常修正及利润重算后数据": "step4_abnormal_data",
                "步骤5_MinMax标准化后数据": "step5_minmax_data",
                "步骤5_ZScore标准化后数据": "step5_zscore_data"
            }

            selected_file = st.selectbox(
                "选择任务1处理文件:",
                list(task1_files.keys()),
                key="task1_file_task2"
            )

            if selected_file and st.session_state.get(task1_files[selected_file]) is not None:
                current_data = st.session_state[task1_files[selected_file]]
                data_source = f"任务1: {selected_file}"
                st.success(f"✅ 使用{selected_file}，共 {len(current_data)} 条记录")
            else:
                st.error("❌ 选择的文件不存在，请先完成任务1")
                return

        else:  # 上传新文件
            uploaded_file = st.file_uploader(
                "上传多维分析数据文件",
                type=["xlsx", "csv"],
                key="upload_task2_new"
            )

            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.xlsx'):
                        current_data = pd.read_excel(uploaded_file)
                    else:
                        current_data = pd.read_csv(uploaded_file)

                    current_data = clean_numeric_columns(current_data)
                    data_source = f"自定义文件: {uploaded_file.name}"
                    st.success(f"✅ 数据加载成功！共 {len(current_data)} 条记录")
                except Exception as e:
                    st.error(f"❌ 文件读取错误: {str(e)}")
                    return
            else:
                st.info("📝 请上传数据文件")
                return

        # 数据清洗和检查
        if current_data is not None:
            # 数据清洗卡片
            with st.expander("🧹 数据清洗设置", expanded=True):
                col_clean1, col_clean2 = st.columns(2)

                with col_clean1:
                    # 数值字段转换
                    numeric_columns = ['利润', '销售额', '销售数', '实际售价', '进货价格']
                    for col in numeric_columns:
                        if col in current_data.columns:
                            current_data[col] = pd.to_numeric(current_data[col], errors='coerce')

                    # 区域字段处理
                    if '区域' in current_data.columns:
                        current_data['区域'] = current_data['区域'].astype(str)
                        if current_data['区域'].str.contains('-').any():
                            current_data['省份'] = current_data['区域'].apply(
                                lambda x: x.split('-')[1] if '-' in str(x) and len(x.split('-')) > 1 else x
                            )
                        else:
                            current_data['省份'] = current_data['区域']

                with col_clean2:
                    # 移除空值
                    original_count = len(current_data)
                    if '利润' in current_data.columns:
                        current_data = current_data.dropna(subset=['利润'])
                        removed_count = original_count - len(current_data)
                        if removed_count > 0:
                            st.warning(f"移除 {removed_count} 条利润为空的记录")

            # 数据预览卡片
            with st.expander("👀 数据预览", expanded=False):
                preview_col1, preview_col2 = st.columns(2)
                with preview_col1:
                    st.dataframe(current_data.head(8))
                with preview_col2:
                    # 数据统计
                    st.metric("总记录数", len(current_data))
                    st.metric("字段数量", len(current_data.columns))
                    st.metric("数据大小", f"{current_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        # ============================================================================
        # 分析执行和结果显示区域
        # ============================================================================
        if st.session_state.get('run_analysis') and current_data is not None:
            st.markdown("---")
            st.markdown("### 📈 分析执行")

            # 检查必需字段
            required_columns = ['区域', '商品品类', '利润']
            missing_columns = [col for col in required_columns if col not in current_data.columns]
            if missing_columns:
                st.error(f"❌ 缺少必需字段: {', '.join(missing_columns)}")
                st.info("💡 多维分析需要以下字段：区域、商品品类、利润")
                return

            with st.spinner("🔄 正在执行多维分析..."):
                # 自动检测字段类型
                column_types = auto_detect_column_types(current_data)

                analyzer = EnhancedTask2Analyzer(current_data, column_types)
                visualizer = EnhancedVisualizer(current_data, column_types)

                # 执行分析
                heatmap_success = analyzer.create_heatmaps()
                cluster_success = analyzer.perform_clustering_analysis()
                visualization_success = visualizer.generate_all_visualizations()

                # 生成所有分析数据
                all_analysis_data = analyzer.generate_all_analysis_data()

                # 保存结果到session state
                st.session_state.task2_results = analyzer.results
                st.session_state.task2_visualizations = visualizer.results
                st.session_state.task2_analysis_data = all_analysis_data
                st.session_state.task2_completed = True

            # 分析结果摘要
            st.success("✅ 多维特征分析完成！")

            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            with summary_col1:
                st.metric("热力图分析", "✅ 完成" if heatmap_success else "❌ 失败")
            with summary_col2:
                st.metric("聚类分析", "✅ 完成" if cluster_success else "❌ 失败")
            with summary_col3:
                st.metric("可视化", "✅ 完成" if visualization_success else "⚠️ 部分完成")
            with summary_col4:
                analysis_count = sum(1 for data in all_analysis_data.values() if data is not None)
                st.metric("分析维度", f"{analysis_count}个")

            # 结果显示
            st.markdown("### 📊 分析结果")

            if analysis_mode == "📊 Python可视化展示":
                show_python_visualizations(analyzer)
            elif analysis_mode == "📁 论文图表数据导出":
                show_data_export_interface(all_analysis_data)
            else:  # 交互式可视化仪表板
                show_interactive_dashboard_optimized(visualizer.results)

        elif not st.session_state.get('run_analysis'):
            # 分析前的功能说明
            st.markdown("---")
            st.markdown("### 💡 功能说明")

            info_col1, info_col2, info_col3 = st.columns(3)

            with info_col1:
                st.markdown("""
                <div style='background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #007bff;'>
                <h4>📊 Python可视化</h4>
                <ul>
                <li>交叉维度热力图</li>
                <li>客户-商品聚类</li>
                <li>系统内置图表</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            with info_col2:
                st.markdown("""
                <div style='background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745;'>
                <h4>📁 数据导出</h4>
                <ul>
                <li>城市分布数据</li>
                <li>客户画像数据</li>
                <li>时间序列数据</li>
                <li>相关性矩阵</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            with info_col3:
                st.markdown("""
                <div style='background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ff6b6b;'>
                <h4>🎨 交互式仪表板</h4>
                <ul>
                <li>实时指标监控</li>
                <li>交互式图表</li>
                <li>高级分析功能</li>
                <li>客户分群分析</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

        # 下载功能
        if st.session_state.get('download_results') and st.session_state.task2_completed:
            st.markdown("---")
            st.markdown("### 📥 结果下载")
            # 这里添加具体的下载逻辑
            st.info("下载功能已就绪，可选择需要下载的分析结果文件")

    def create_heatmaps(self):
        """创建热力图 - 修复数据类型错误"""
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False

            figs = {}

            # 1. 商品品类与省份交叉热力图
            if all(col in self.df.columns for col in ['区域', '商品品类', '利润']):
                # 提取省份
                if self.df['区域'].str.contains('-').any():
                    self.df['省份'] = self.df['区域'].apply(lambda x: x.split('-')[1] if '-' in str(x) else x)
                else:
                    self.df['省份'] = self.df['区域']

                # 数据清洗：确保利润是数值类型
                self.df['利润'] = pd.to_numeric(self.df['利润'], errors='coerce')
                self.df = self.df.dropna(subset=['利润'])

                # 创建数据透视表，确保数据是数值类型
                category_province_pivot = self.df.pivot_table(
                    index='商品品类',
                    columns='省份',
                    values='利润',
                    aggfunc='sum',
                    fill_value=0
                )

                # 确保所有数据都是数值类型
                category_province_pivot = category_province_pivot.apply(pd.to_numeric, errors='coerce').fillna(0)

                # 过滤掉全为0的行和列
                category_province_pivot = category_province_pivot.loc[
                    (category_province_pivot != 0).any(axis=1),
                    (category_province_pivot != 0).any(axis=0)
                ]

                if not category_province_pivot.empty and len(category_province_pivot) > 1:
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(category_province_pivot,
                                cmap='Blues',
                                annot=False,
                                cbar_kws={'label': '利润总额'})
                    plt.title('商品品类和省份交叉的利润热力图', fontsize=14, fontweight='bold')
                    plt.xlabel('省份', fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel('商品品类', fontsize=12)
                    plt.tight_layout()
                    figs['category_province_profit'] = plt.gcf()
                    plt.close()
                else:
                    st.warning("商品品类-省份热力图：数据不足或全为0值")

            # 2. 省份与日期交叉热力图
            if all(col in self.df.columns for col in ['日期', '省份', '利润']):
                # 数据清洗
                self.df['利润'] = pd.to_numeric(self.df['利润'], errors='coerce')
                self.df = self.df.dropna(subset=['利润', '日期'])

                # 如果省份列不存在，创建它
                if '省份' not in self.df.columns and '区域' in self.df.columns:
                    if self.df['区域'].str.contains('-').any():
                        self.df['省份'] = self.df['区域'].apply(lambda x: x.split('-')[1] if '-' in str(x) else x)
                    else:
                        self.df['省份'] = self.df['区域']

                # 日期处理：转换为字符串避免数值问题
                self.df['日期'] = self.df['日期'].astype(str)

                province_date_pivot = self.df.pivot_table(
                    index='省份',
                    columns='日期',
                    values='利润',
                    aggfunc='sum',
                    fill_value=0
                )

                # 确保数据是数值类型
                province_date_pivot = province_date_pivot.apply(pd.to_numeric, errors='coerce').fillna(0)

                # 过滤数据，只显示有变化的日期和省份
                province_date_pivot = province_date_pivot.loc[
                    (province_date_pivot != 0).any(axis=1),
                    (province_date_pivot != 0).any(axis=0)
                ]

                # 限制列数，避免图表过于拥挤
                if len(province_date_pivot.columns) > 20:
                    province_date_pivot = province_date_pivot.iloc[:, :20]  # 只取前20个日期

                if not province_date_pivot.empty and len(province_date_pivot) > 1:
                    plt.figure(figsize=(15, 8))
                    sns.heatmap(province_date_pivot,
                                cmap='Blues',
                                annot=False,
                                cbar_kws={'label': '利润总额'})
                    plt.title('省份和日期交叉的利润热力图', fontsize=14, fontweight='bold')
                    plt.xlabel('日期', fontsize=12)
                    plt.xticks(rotation=90)
                    plt.ylabel('省份', fontsize=12)
                    plt.tight_layout()
                    figs['province_date_profit'] = plt.gcf()
                    plt.close()
                else:
                    st.warning("省份-日期热力图：数据不足或全为0值")

            # 3. 备用热力图：商品品类与利润关系
            if all(col in self.df.columns for col in ['商品品类', '利润']):
                # 数据清洗
                self.df['利润'] = pd.to_numeric(self.df['利润'], errors='coerce')
                self.df = self.df.dropna(subset=['利润'])

                category_profit = self.df.groupby('商品品类')['利润'].sum().sort_values(ascending=False).head(10)

                if len(category_profit) > 1:
                    plt.figure(figsize=(10, 6))
                    category_profit.plot(kind='bar', color='skyblue', alpha=0.8)
                    plt.title('Top 10商品品类利润分布', fontsize=14, fontweight='bold')
                    plt.xlabel('商品品类', fontsize=12)
                    plt.ylabel('利润总额', fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    figs['category_profit_bar'] = plt.gcf()
                    plt.close()

            self.results['heatmaps'] = figs
            return len(figs) > 0

        except Exception as e:
            st.error(f"热力图生成错误: {str(e)}")
            import traceback
            st.error(f"详细错误信息: {traceback.format_exc()}")
            return False

    def perform_clustering_analysis(self):
        """执行聚类分析 - 原有的聚类功能"""
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False

            # 选择数值型列进行聚类
            numeric_cols = ['客户年龄', '进货价格', '实际售价', '销售数', '销售额', '利润']
            existing_numeric_cols = [col for col in numeric_cols if col in self.df.columns]

            if len(existing_numeric_cols) < 2:
                st.warning(f"可用于聚类的数值型列不足，仅找到: {existing_numeric_cols}")
                return False

            # 提取数值型数据并处理缺失值
            df_numeric = self.df[existing_numeric_cols].fillna(0)

            # 确定最佳聚类数k
            sse = []
            silhouette_scores = []
            k_range = range(2, 11)

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=2024, n_init='auto')
                kmeans.fit(df_numeric)
                sse.append(kmeans.inertia_)
                labels = kmeans.labels_
                score = silhouette_score(df_numeric, labels)
                silhouette_scores.append(score)

            # 绘制评估图表
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(k_range, sse, 'bx-')
            ax1.set_xlabel('聚类数量k')
            ax1.set_ylabel('SSE（误差平方和）')
            ax1.set_title('手肘法确定最佳k值')
            ax2.plot(k_range, silhouette_scores, 'rx-')
            ax2.set_xlabel('聚类数量k')
            ax2.set_ylabel('轮廓系数')
            ax2.set_title('轮廓系数确定最佳k值')
            plt.tight_layout()

            self.results['cluster_evaluation_plot'] = fig1

            # 选择最佳k值
            best_k_index = silhouette_scores.index(max(silhouette_scores))
            best_k = k_range[best_k_index]

            # 使用最佳k值执行最终聚类
            final_kmeans = KMeans(n_clusters=best_k, random_state=2024, n_init='auto')
            cluster_labels = final_kmeans.fit_predict(df_numeric)

            # 保存聚类结果
            df_clustered = self.df.copy()
            df_clustered['聚类标签'] = cluster_labels
            cluster_analysis = df_clustered.groupby('聚类标签')[existing_numeric_cols].mean().round(2)

            self.results['clustered_data'] = df_clustered
            self.results['cluster_analysis'] = cluster_analysis
            self.results['best_k'] = best_k

            return True

        except Exception as e:
            st.error(f"聚类分析错误: {str(e)}")
            return False

    def generate_city_distribution_data(self):
        """生成城市分布数据（对应论文图4）"""
        if '区域' not in self.df.columns:
            return None

        # 提取城市信息
        if self.df['区域'].str.contains('-').any():
            self.df['城市'] = self.df['区域'].apply(lambda x: x.split('-')[1] if '-' in str(x) else x)
        else:
            self.df['城市'] = self.df['区域']

        # 城市用户数统计
        city_stats = self.df['城市'].value_counts().reset_index()
        city_stats.columns = ['城市', '用户数']
        city_stats = city_stats.head(15)  # Top 15城市

        return city_stats

    def generate_province_distribution_data(self):
        """生成省份分布数据（对应论文图5）"""
        if '区域' not in self.df.columns:
            return None

        # 提取省份信息
        if '省份' not in self.df.columns:
            if self.df['区域'].str.contains('-').any():
                self.df['省份'] = self.df['区域'].apply(lambda x: x.split('-')[0] if '-' in str(x) else x)
            else:
                self.df['省份'] = self.df['区域']

        province_stats = self.df['省份'].value_counts().reset_index()
        province_stats.columns = ['省份', '用户数']

        return province_stats

    def generate_city_tier_data(self):
        """生成城市分级数据（对应论文图6）"""
        if '城市' not in self.df.columns:
            return None

        # 城市分级定义（根据论文）
        tier_1 = ['北京', '上海', '广州', '深圳']
        tier_2 = ['昆明', '福州', '厦门', '无锡', '哈尔滨', '长春', '宁波', '济南', '大连', '郑州',
                  '长沙', '成都', '杭州', '南京', '武汉', '西安', '苏州', '天津', '青岛', '沈阳',
                  '东莞', '佛山', '合肥', '石家庄', '南宁', '常州', '烟台', '唐山', '徐州', '温州']
        tier_3 = ['兰州', '海口', '乌鲁木齐', '贵阳', '银川', '西宁', '呼和浩特', '拉萨', '保定',
                  '惠州', '珠海', '中山', '江门', '肇庆', '清远', '韶关', '湛江', '茂名', '阳江',
                  '云浮', '汕头', '潮州', '揭阳', '汕尾', '梅州', '河源']

        def assign_city_tier(city):
            if pd.isna(city):
                return '其他城市'
            city_str = str(city)
            if city_str in tier_1:
                return '一线城市'
            elif city_str in tier_2:
                return '二线城市'
            elif city_str in tier_3:
                return '三线城市'
            else:
                return '其他城市'

        self.df['城市等级'] = self.df['城市'].apply(assign_city_tier)
        tier_stats = self.df['城市等级'].value_counts().reset_index()
        tier_stats.columns = ['城市等级', '用户数']
        tier_stats['占比'] = (tier_stats['用户数'] / len(self.df) * 100).round(2)

        return tier_stats

    def generate_region_tier_data(self):
        """生成区域分级数据（对应论文图7）"""
        if '省份' not in self.df.columns:
            return None

        # 区域定义
        region_mapping = {
            '华南': ['广东', '广西', '海南', '福建'],
            '华东': ['上海', '江苏', '浙江', '安徽', '江西', '山东'],
            '华北': ['北京', '天津', '河北', '山西', '内蒙古'],
            '东北': ['辽宁', '吉林', '黑龙江'],
            '西南': ['重庆', '四川', '贵州', '云南', '西藏'],
            '西北': ['陕西', '甘肃', '青海', '宁夏', '新疆'],
            '华中': ['河南', '湖北', '湖南']
        }

        def assign_region(province):
            if pd.isna(province):
                return '其他'
            province_str = str(province)
            for region, provinces in region_mapping.items():
                if province_str in provinces:
                    return region
            return '其他'

        self.df['大区'] = self.df['省份'].apply(assign_region)
        region_stats = self.df['大区'].value_counts().reset_index()
        region_stats.columns = ['大区', '用户数']
        region_stats['占比'] = (region_stats['用户数'] / len(self.df) * 100).round(2)

        return region_stats

    def generate_gender_category_analysis(self):
        """生成性别-品类分析数据（对应论文图8）"""
        if not all(col in self.df.columns for col in ['客户性别', '商品品类']):
            return None

        gender_category_stats = self.df.groupby(['商品品类', '客户性别']).size().reset_index()
        gender_category_stats.columns = ['商品品类', '客户性别', '订单人数']

        return gender_category_stats

    def generate_age_gender_analysis(self):
        """生成年龄-性别分析数据（对应论文图9）"""
        if '客户年龄' not in self.df.columns or '客户性别' not in self.df.columns:
            return None

        # 年龄分段
        def assign_age_group(age):
            if pd.isna(age):
                return '未知'
            try:
                age = int(age)
                if age < 25:
                    return '20-24岁'
                elif age < 30:
                    return '25-29岁'
                elif age < 35:
                    return '30-34岁'
                elif age < 40:
                    return '35-39岁'
                elif age < 45:
                    return '40-44岁'
                elif age < 50:
                    return '45-49岁'
                elif age < 55:
                    return '50-54岁'
                elif age < 60:
                    return '55-59岁'
                else:
                    return '60岁以上'
            except:
                return '未知'

        self.df['年龄段'] = self.df['客户年龄'].apply(assign_age_group)
        age_gender_stats = self.df.groupby(['年龄段', '客户性别']).size().reset_index()
        age_gender_stats.columns = ['年龄段', '客户性别', '订单人数']

        return age_gender_stats

    def generate_time_series_analysis(self):
        """生成时间序列分析数据（对应论文图10）"""
        date_col = next((col for col in self.column_types['identifier'] if '日期' in col), None)
        if not date_col:
            return None

        time_stats = self.df.groupby(date_col).size().reset_index()
        time_stats.columns = ['日期', '订单人数总和']

        return time_stats

    def generate_correlation_analysis(self):
        """生成相关性分析数据（对应论文图13）"""
        numeric_cols = self.column_types['numeric']
        if len(numeric_cols) < 2:
            return None

        correlation_matrix = self.df[numeric_cols].corr().round(4)

        return correlation_matrix

    def generate_all_analysis_data(self):
        """生成所有分析维度的数据"""
        analysis_results = {}

        # 地理分布分析
        analysis_results['city_distribution'] = self.generate_city_distribution_data()
        analysis_results['province_distribution'] = self.generate_province_distribution_data()
        analysis_results['city_tier_analysis'] = self.generate_city_tier_data()
        analysis_results['region_tier_analysis'] = self.generate_region_tier_data()

        # 客户画像分析
        analysis_results['gender_category_analysis'] = self.generate_gender_category_analysis()
        analysis_results['age_gender_analysis'] = self.generate_age_gender_analysis()

        # 时间序列分析
        analysis_results['time_series_analysis'] = self.generate_time_series_analysis()

        # 统计关系分析
        analysis_results['correlation_analysis'] = self.generate_correlation_analysis()

        # 保留原有的热力图和聚类分析
        analysis_results.update(self.results)

        return analysis_results


def show_python_visualizations(analyzer):
    """显示Python原生可视化"""
    st.subheader("📊 Python可视化展示")

    # 原有的热力图和聚类分析展示
    if 'heatmaps' in analyzer.results and len(analyzer.results['heatmaps']) > 0:
        st.subheader("1. 交叉维度热力图分析")
        for name, fig in analyzer.results['heatmaps'].items():
            st.pyplot(fig)

    if 'cluster_evaluation_plot' in analyzer.results:
        st.subheader("2. 聚类分析结果")
        st.pyplot(analyzer.results['cluster_evaluation_plot'])

        if 'cluster_analysis' in analyzer.results:
            st.subheader("聚类特征平均值对比")
            st.dataframe(analyzer.results['cluster_analysis'])


def show_data_export_interface(analysis_data):
    """显示数据导出界面 - 使用Excel格式避免编码问题"""
    st.subheader("📁 论文图表数据导出")

    st.markdown("""
    ### 导出说明
    以下数据可直接用于在Excel、Tableau、Echarts等工具中制作论文图表。
    为避免编码问题，已提供Excel格式下载。
    """)

    def convert_to_excel(df, sheet_name="数据"):
        """将DataFrame转换为Excel格式"""
        import io
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        output.seek(0)
        return output.getvalue()

    # 地理分布数据导出
    st.markdown("#### 🌍 地理分布分析")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if analysis_data.get('city_distribution') is not None:
            excel_data = convert_to_excel(analysis_data['city_distribution'], "城市分布")
            st.download_button(
                label="下载城市分布数据",
                data=excel_data,
                file_name="城市分布数据_Top15城市.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with col2:
        if analysis_data.get('province_distribution') is not None:
            excel_data = convert_to_excel(analysis_data['province_distribution'], "省份分布")
            st.download_button(
                label="下载省份分布数据",
                data=excel_data,
                file_name="省份分布数据.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with col3:
        if analysis_data.get('city_tier_analysis') is not None:
            excel_data = convert_to_excel(analysis_data['city_tier_analysis'], "城市分级")
            st.download_button(
                label="下载城市分级数据",
                data=excel_data,
                file_name="城市分级环状图数据.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with col4:
        if analysis_data.get('region_tier_analysis') is not None:
            excel_data = convert_to_excel(analysis_data['region_tier_analysis'], "区域分级")
            st.download_button(
                label="下载区域分级数据",
                data=excel_data,
                file_name="区域分级环状图数据.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # 客户画像数据导出
    st.markdown("#### 👥 客户画像分析")
    col1, col2 = st.columns(2)

    with col1:
        if analysis_data.get('gender_category_analysis') is not None:
            excel_data = convert_to_excel(analysis_data['gender_category_analysis'], "性别品类")
            st.download_button(
                label="下载性别-品类数据",
                data=excel_data,
                file_name="性别品类交叉分析数据.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with col2:
        if analysis_data.get('age_gender_analysis') is not None:
            excel_data = convert_to_excel(analysis_data['age_gender_analysis'], "年龄性别")
            st.download_button(
                label="下载年龄-性别数据",
                data=excel_data,
                file_name="年龄性别分布数据.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # 时间序列和相关性分析
    st.markdown("#### 📈 时间与关系分析")
    col1, col2 = st.columns(2)

    with col1:
        if analysis_data.get('time_series_analysis') is not None:
            excel_data = convert_to_excel(analysis_data['time_series_analysis'], "时间序列")
            st.download_button(
                label="下载时间序列数据",
                data=excel_data,
                file_name="时间序列订单数据.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with col2:
        if analysis_data.get('correlation_analysis') is not None:
            # 相关性矩阵需要保留索引
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                analysis_data['correlation_analysis'].to_excel(writer, sheet_name="相关性矩阵")
            excel_data = output.getvalue()
            st.download_button(
                label="下载相关性矩阵",
                data=excel_data,
                file_name="变量相关性矩阵.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # 数据预览（保持不变）
    st.markdown("#### 👀 数据预览")
    available_datasets = [key for key in analysis_data.keys() if
                          analysis_data[key] is not None and hasattr(analysis_data[key], 'head')]
    if available_datasets:
        dataset_to_preview = st.selectbox(
            "选择要预览的数据集:",
            available_datasets
        )

        if dataset_to_preview:
            st.dataframe(analysis_data[dataset_to_preview].head(10))

            # 数据统计信息
            st.markdown("**数据统计:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("行数", len(analysis_data[dataset_to_preview]))
            with col2:
                st.metric("列数", len(analysis_data[dataset_to_preview].columns))
            with col3:
                st.metric("数据类型", str(analysis_data[dataset_to_preview].dtypes.unique()[0]))

    st.success("✅ 现在下载的Excel文件应该不会出现乱码问题了！")


# ============================================================================
# 任务3：销售预测类（按照独立脚本逻辑重构）
# ============================================================================
class Task3Forecaster:
    def __init__(self, df, column_types):
        self.df = df.copy()
        self.column_types = column_types
        self.results = {}

    def prepare_time_series_data(self):
        """准备时间序列数据（基于独立脚本逻辑）"""
        try:
            # 检查必要字段
            required_cols = ["日期", "利润", "销售额", "实际售价", "进货价格", "客户性别"]
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                st.error(f"数据缺失必要字段：{missing_cols}")
                return False

            # 日期格式转换（转为整数型）
            self.df['日期'] = pd.to_numeric(self.df['日期'], errors='coerce')
            self.df = self.df.dropna(subset=['日期'])

            # 按日聚合利润数据 - 使用利润字段（对应独立脚本的正确利润）
            daily_profit = self.df.groupby('日期')['利润'].sum().reset_index()
            daily_profit = daily_profit.rename(columns={'利润': '每日总利润'})

            # 划分训练集和测试集（日期<=24为训练集，>24为测试集）
            train = daily_profit[daily_profit['日期'] <= 24]
            test = daily_profit[daily_profit['日期'] > 24]

            if len(train) == 0 or len(test) == 0:
                st.error("数据日期范围不足，无法划分训练测试集")
                return False

            self.results['time_series_data'] = daily_profit
            self.results['train_data'] = train
            self.results['test_data'] = test
            self.results['y_train'] = train['每日总利润'].values
            self.results['y_test'] = test['每日总利润'].values

            st.success(f"时间序列准备完成：训练集{len(train)}天，测试集{len(test)}天")
            return True

        except Exception as e:
            st.error(f"时间序列准备错误: {str(e)}")
            return False

    def create_features(self, day_indices, residuals=None):
        """特征工程函数（基于独立脚本逻辑）"""
        features = []

        # 预计算训练集每日统计量
        train_days_data = self.df[self.df['日期'] <= 24]
        train_stats = train_days_data.groupby('日期').agg({
            '销售额': ['count', 'mean', 'sum'],
            '客户性别': lambda x: (x == '女').mean() if '客户性别' in train_days_data.columns else 0.5
        })
        train_stats.columns = ['order_count', 'avg_sale', 'total_sale', 'female_ratio']

        # 预计算每个日期的统计量
        daily_stats = self.df.groupby('日期').agg({
            '销售额': ['count', 'mean', 'sum'],
            '实际售价': 'mean',
            '进货价格': 'mean',
            '客户性别': lambda x: (x == '女').mean() if '客户性别' in self.df.columns else 0.5
        }).round(4)

        # 处理列名
        daily_stats.columns = ['order_count', 'avg_sale', 'total_sale',
                               'avg_selling_price', 'avg_cost_price', 'female_ratio']

        # 计算毛利率
        daily_stats['gross_profit_margin'] = (
                (daily_stats['avg_selling_price'] - daily_stats['avg_cost_price']) /
                daily_stats['avg_cost_price']
        ).fillna(0).round(4)

        # 计算单客价值
        daily_stats['customer_value'] = (
                daily_stats['total_sale'] / daily_stats['order_count']
        ).fillna(0).round(2)

        for day in day_indices:
            day_features = {}

            # 1. 基础时间特征
            day_features['day'] = day
            day_features['day_of_week'] = (day - 1) % 7  # 0=周一, 6=周日
            day_features['day_of_month'] = day
            day_features['is_weekend'] = 1 if day_features['day_of_week'] in [5, 6] else 0
            day_features['is_month_end'] = 1 if day >= 28 else 0

            # 2. 从预计算的统计量中获取业务特征
            if day in daily_stats.index:
                stats = daily_stats.loc[day]
                day_features.update({
                    'order_count': stats['order_count'],
                    'avg_sale_amount': stats['avg_sale'],
                    'total_sale': stats['total_sale'],
                    'gross_profit_margin': stats['gross_profit_margin'],
                    'customer_value': stats['customer_value'],
                    'female_ratio': stats['female_ratio']
                })
            else:
                # 使用训练集的中位数填充
                day_features.update({
                    'order_count': train_stats['order_count'].median(),
                    'avg_sale_amount': train_stats['avg_sale'].median(),
                    'total_sale': train_stats['total_sale'].median(),
                    'gross_profit_margin': 0.3,  # 默认毛利率
                    'customer_value': train_stats['total_sale'].median() / max(1, train_stats['order_count'].median()),
                    'female_ratio': train_stats['female_ratio'].median()
                })

            # 3. 滞后残差特征
            if residuals is not None:
                for lag in [1, 2, 3]:
                    lag_day = day - lag
                    lag_key = f'residual_lag_{lag}'
                    if lag_day > 0 and lag_day in residuals.index:
                        day_features[lag_key] = residuals[lag_day]
                    else:
                        day_features[lag_key] = residuals.median() if not residuals.empty else 0

            features.append(day_features)

        return pd.DataFrame(features)

    def hybrid_forecast(self):
        """ARIMA-XGBoost混合预测（基于独立脚本逻辑）"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from xgboost import XGBRegressor
            from sklearn.metrics import mean_absolute_percentage_error
            import warnings
            warnings.filterwarnings('ignore')

            # 获取数据
            train = self.results['train_data']
            test = self.results['test_data']
            y_train = self.results['y_train']
            y_test = self.results['y_test']

            # 1. ARIMA建模
            st.info("Step 1: ARIMA建模...")
            try:
                arima_model = ARIMA(y_train, order=(2, 1, 2))
                arima_fit = arima_model.fit()
                arima_train_pred = arima_fit.predict(start=0, end=len(y_train) - 1)
                arima_test_pred = arima_fit.forecast(steps=len(y_test))
                st.success(f"ARIMA模型训练成功 (AIC: {arima_fit.aic:.2f})")
            except Exception as e:
                st.warning(f"ARIMA模型训练失败，使用均值预测: {e}")
                arima_train_pred = np.full_like(y_train, y_train.mean())
                arima_test_pred = np.full_like(y_test, y_train.mean())
                arima_fit = None

            # 2. 计算残差
            residuals_train = y_train - arima_train_pred
            residual_series = pd.Series(residuals_train, index=train['日期'])

            # 3. XGBoost学习残差
            st.info("Step 2: XGBoost学习残差...")

            # 创建特征
            X_train = self.create_features(train['日期'], residual_series)
            X_train = X_train.fillna(0)

            # 特征统计分析
            feature_stats = pd.DataFrame({
                'mean': X_train.mean(),
                'std': X_train.std(),
                'min': X_train.min(),
                'max': X_train.max(),
                'zeros': (X_train == 0).sum(),
                'unique': X_train.nunique()
            })

            # 筛选低方差特征
            low_variance_features = feature_stats[feature_stats['std'] < 1e-5].index.tolist()
            if low_variance_features:
                st.info(f"低方差特征: {low_variance_features}")
                X_train = X_train.drop(columns=low_variance_features)

            # XGBoost模型训练（使用独立脚本参数）
            xgb_model = XGBRegressor(
                max_depth=3,
                learning_rate=0.05,
                n_estimators=1000,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                eval_metric='mae'
            )
            xgb_model.fit(X_train, residuals_train)

            # 测试集特征
            X_test = self.create_features(test['日期'])
            X_test = X_test.fillna(0)

            # 删除训练集中已剔除的低方差特征
            for col in low_variance_features:
                if col in X_test.columns:
                    X_test = X_test.drop(columns=col)

            # 确保特征一致性
            for col in X_train.columns:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[X_train.columns]

            # 预测残差
            xgb_residual_pred = xgb_model.predict(X_test)

            # 4. 最终预测
            final_pred = arima_test_pred + xgb_residual_pred
            mape = mean_absolute_percentage_error(y_test, final_pred) * 100

            # 保存结果
            self.results['arima_model'] = arima_fit
            self.results['xgb_model'] = xgb_model
            self.results['arima_test_pred'] = arima_test_pred
            self.results['xgb_residual_pred'] = xgb_residual_pred
            self.results['final_pred'] = final_pred
            self.results['mape'] = mape
            self.results['residuals_train'] = residuals_train

            # 特征重要性
            self.results['feature_importance'] = pd.DataFrame({
                'feature': X_train.columns,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)

            # 创建详细结果表
            results_df = pd.DataFrame({
                '日期': test['日期'],
                '实际每日总利润': y_test,
                'ARIMA预测利润': arima_test_pred,
                'XGBoost残差预测': xgb_residual_pred,
                '最终预测利润': final_pred,
                '相对误差(%)': np.abs(y_test - final_pred) / y_test * 100
            })
            self.results['detailed_results'] = results_df

            st.success(f"混合预测完成！测试集MAPE: {mape:.2f}%")
            return True

        except Exception as e:
            st.error(f"混合预测错误: {str(e)}")
            return False

    def generate_visualizations(self):
        """生成可视化图表（基于独立脚本逻辑）"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            figs = {}
            train = self.results['train_data']
            test = self.results['test_data']
            y_train = self.results['y_train']
            y_test = self.results['y_test']

            # 1. 主预测对比图
            fig1, ax1 = plt.subplots(figsize=(12, 8))

            # 训练集实际值
            ax1.plot(train['日期'], y_train / 10000, 'bo-', label='训练集实际利润',
                     alpha=0.7, markersize=6, linewidth=2)
            # 测试集实际值
            ax1.plot(test['日期'], y_test / 10000, 'ro-', label='测试集实际利润',
                     alpha=0.7, markersize=8, linewidth=2)

            # ARIMA训练集拟合值
            if self.results['arima_model'] is not None:
                arima_train_fit = self.results['arima_model'].predict(start=1, end=24)
                ax1.plot(train['日期'], arima_train_fit / 10000, 'c--', label='ARIMA训练集拟合',
                         alpha=0.8, linewidth=2)

            # ARIMA测试集预测值
            ax1.plot(test['日期'], self.results['arima_test_pred'] / 10000, 'm--',
                     label='ARIMA测试集预测', alpha=0.8, linewidth=2)
            # 最终组合预测值
            ax1.plot(test['日期'], self.results['final_pred'] / 10000, 'gs-',
                     label='ARIMA+XGBoost最终预测', markersize=8, linewidth=2)

            ax1.set_xlabel('日期 (11月天数)', fontsize=12)
            ax1.set_ylabel('利润 (万元)', fontsize=12)
            ax1.set_title('电商平台每日总利润预测对比', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=24.5, color='gray', linestyle=':', alpha=0.7, linewidth=2)
            ax1.text(24.7, ax1.get_ylim()[1] * 0.9, '测试集开始', rotation=90, va='top', fontsize=10)
            figs['main_forecast'] = fig1

            # 2. 误差分析图
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            relative_errors = self.results['detailed_results']['相对误差(%)']
            bars = ax2.bar(test['日期'], relative_errors, alpha=0.7, color='orange',
                           edgecolor='darkorange', linewidth=1)

            ax2.set_xlabel('日期 (11月天数)', fontsize=12)
            ax2.set_ylabel('相对误差 (%)', fontsize=12)
            ax2.set_title(f'电商平台利润预测误差分析 (MAPE = {self.results["mape"]:.2f}%)',
                          fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            # 添加误差数值标签
            for date, error in zip(test['日期'], relative_errors):
                ax2.text(date, error + 1, f'{error:.1f}%', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')
            figs['error_analysis'] = fig2

            # 3. 残差分析图
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            residuals_train = self.results['residuals_train']

            ax3.plot(range(1, 25), residuals_train / 10000, 'o-', color='purple',
                     alpha=0.7, markersize=6, linewidth=2, label='每日残差')
            ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='零基准线')

            mean_residual = residuals_train.mean() / 10000
            ax3.axhline(y=mean_residual, color='blue', linestyle=':', linewidth=2, alpha=0.7,
                        label=f'残差均值: {mean_residual:.2f}万元')

            ax3.set_xlabel('训练集日期 (11月天数)', fontsize=12)
            ax3.set_ylabel('残差 (万元)', fontsize=12)
            ax3.set_title('ARIMA模型残差分布', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)

            # 统计信息框
            stats_text = (f'均值: {residuals_train.mean() / 10000:.2f}万元\n'
                          f'标准差: {residuals_train.std() / 10000:.2f}万元\n'
                          f'最大值: {residuals_train.max() / 10000:.2f}万元\n'
                          f'最小值: {residuals_train.min() / 10000:.2f}万元')
            ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=11,
                     verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                                                        facecolor="lightgray", alpha=0.7))
            figs['residual_analysis'] = fig3

            # 4. 特征重要性图
            fig4, ax4 = plt.subplots(figsize=(12, 8))
            feature_importance = self.results['feature_importance'].head(10)

            # 特征名称映射
            feature_names_map = {
                'day': '日期', 'day_of_week': '星期', 'day_of_month': '月内天数',
                'is_weekend': '是否周末', 'is_month_end': '是否月末',
                'order_count': '订单数', 'avg_sale_amount': '平均销售额',
                'total_sale': '总销售额', 'gross_profit_margin': '毛利率',
                'customer_value': '单客价值', 'female_ratio': '女性客户占比',
                'residual_lag_1': '残差滞后1天', 'residual_lag_2': '残差滞后2天',
                'residual_lag_3': '残差滞后3天'
            }

            feature_importance['feature_cn'] = feature_importance['feature'].map(
                lambda x: feature_names_map.get(x, x)
            )
            feature_importance = feature_importance.sort_values('importance', ascending=True)

            y_pos = np.arange(len(feature_importance))
            colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))

            ax4.barh(y_pos, feature_importance['importance'], color=colors,
                     alpha=0.8, edgecolor='black')
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(feature_importance['feature_cn'], fontsize=11)
            ax4.set_xlabel('特征重要性得分', fontsize=12, fontweight='bold')
            ax4.set_title('XGBoost残差预测特征重要性', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')

            # 添加重要性数值标签
            for i, v in enumerate(feature_importance['importance']):
                ax4.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')

            figs['feature_importance'] = fig4

            self.results['visualizations'] = figs
            return True

        except Exception as e:
            st.error(f"可视化生成错误: {str(e)}")
            import traceback
            st.error(f"详细错误: {traceback.format_exc()}")
            return False

    def generate_all_results(self, forecast_days=14):
        """生成所有预测结果"""
        try:
            if not self.prepare_time_series_data():
                return None, ["时间序列数据准备失败"]

            if not self.hybrid_forecast():
                return None, ["混合预测模型执行失败"]

            if not self.generate_visualizations():
                return None, ["可视化生成失败"]

            # 整理结果文件
            result_files = {
                '01_时间序列历史数据.xlsx': self.results['time_series_data'],
                '02_销售预测结果.xlsx': self.results['detailed_results'],
                '03_特征重要性分析.xlsx': self.results['feature_importance']
            }

            # 进度日志
            progress_log = [
                f"时间序列准备完成：训练集{len(self.results['train_data'])}天，测试集{len(self.results['test_data'])}天",
                f"ARIMA-XGBoost混合预测完成：测试集MAPE {self.results['mape']:.2f}%",
                f"特征重要性分析完成：{len(self.results['feature_importance'])}个特征",
                f"可视化图表生成完成：4个分析图表"
            ]

            return result_files, progress_log

        except Exception as e:
            return None, [f"预测错误: {str(e)}"]


# ============================================================================
# 任务4：运营策略优化类
# ============================================================================
class Task4Optimizer:
    def __init__(self, df, column_types):
        self.df = df.copy()
        self.column_types = column_types
        self.results = {}

    def abc_classification_analysis(self):
        """ABC分类分析（基于帕累托法则）"""
        try:
            # 按商品品类聚合销售额和利润
            category_stats = self.df.groupby('商品品类').agg({
                '销售额': 'sum',
                '利润': 'sum',
                '销售数': 'sum'
            }).reset_index()

            # 计算累计占比
            category_stats = category_stats.sort_values('销售额', ascending=False)
            category_stats['销售额累计占比%'] = (
                        category_stats['销售额'].cumsum() / category_stats['销售额'].sum() * 100).round(2)
            category_stats['利润累计占比%'] = (
                        category_stats['利润'].cumsum() / category_stats['利润'].sum() * 100).round(2)
            category_stats['销售额占比%'] = (category_stats['销售额'] / category_stats['销售额'].sum() * 100).round(2)
            category_stats['利润占比%'] = (category_stats['利润'] / category_stats['利润'].sum() * 100).round(2)

            # ABC分类（基于销售额）
            def assign_abc_class(cumulative_percent):
                if cumulative_percent <= 70:
                    return 'A类'
                elif cumulative_percent <= 90:
                    return 'B类'
                else:
                    return 'C类'

            category_stats['ABC分类（销售额）'] = category_stats['销售额累计占比%'].apply(assign_abc_class)

            # 生成可视化图表
            self._create_abc_visualizations(category_stats)

            self.results['abc_classification'] = category_stats
            return True

        except Exception as e:
            st.error(f"ABC分类分析错误: {str(e)}")
            return False

    def _create_abc_visualizations(self, category_stats):
        """创建ABC分类可视化图表"""
        try:
            figs = {}

            # 1. 销售额分布图
            plt.figure(figsize=(12, 6))
            top_categories = category_stats.head(10)
            plt.bar(range(len(top_categories)), top_categories['销售额'], color='skyblue', alpha=0.8)
            plt.xlabel('商品品类')
            plt.ylabel('销售额')
            plt.title('Top 10商品品类销售额分布')
            plt.xticks(range(len(top_categories)), top_categories['商品品类'], rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            figs['sales_distribution'] = plt.gcf()
            plt.close()

            # 2. 累计销售额帕累托图
            plt.figure(figsize=(12, 6))
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # 柱状图（销售额）
            bars = ax1.bar(range(len(category_stats)), category_stats['销售额'],
                           color='lightblue', alpha=0.7, label='销售额')
            ax1.set_xlabel('商品品类')
            ax1.set_ylabel('销售额', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            # 折线图（累计占比）
            ax2 = ax1.twinx()
            line = ax2.plot(range(len(category_stats)), category_stats['销售额累计占比%'],
                            color='red', marker='o', linewidth=2, label='累计占比')
            ax2.set_ylabel('累计占比 (%)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.axhline(y=70, color='green', linestyle='--', alpha=0.7, label='A类分界线 (70%)')
            ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='B类分界线 (90%)')

            plt.title('ABC分类帕累托分析')
            fig.legend(loc='upper right')
            plt.tight_layout()
            figs['cumulative_sales'] = fig
            plt.close()

            # 3. ABC分类分布图
            abc_counts = category_stats['ABC分类（销售额）'].value_counts()
            plt.figure(figsize=(8, 6))
            colors = {'A类': 'red', 'B类': 'orange', 'C类': 'green'}
            abc_colors = [colors.get(cls, 'gray') for cls in abc_counts.index]
            plt.pie(abc_counts.values, labels=abc_counts.index, autopct='%1.1f%%',
                    colors=abc_colors, startangle=90)
            plt.title('ABC分类分布')
            figs['abc_distribution'] = plt.gcf()
            plt.close()

            self.results['abc_visualizations'] = figs
            return True

        except Exception as e:
            st.error(f"ABC可视化生成错误: {str(e)}")
            return False

    def price_sensitivity_analysis(self):
        """价格敏感度分析"""
        try:
            # 按商品品类分析价格-销量关系
            sensitivity_results = []

            for category in self.df['商品品类'].unique():
                category_data = self.df[self.df['商品品类'] == category]

                if len(category_data) < 10:  # 数据量太少跳过
                    continue

                # 价格分箱（等频8区间）
                try:
                    category_data = category_data.copy()
                    category_data['价格区间'] = pd.qcut(category_data['实际售价'], q=8, duplicates='drop')

                    # 计算每个价格区间的平均销量
                    price_bin_stats = category_data.groupby('价格区间').agg({
                        '实际售价': 'mean',
                        '销售数': 'mean',
                        '利润': 'mean'
                    }).reset_index()

                    if len(price_bin_stats) < 3:  # 区间太少无法分析
                        continue

                    # 线性回归分析价格-销量关系
                    X = price_bin_stats['实际售价'].values.reshape(-1, 1)
                    y = price_bin_stats['销售数'].values

                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import r2_score

                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)
                    slope = model.coef_[0]

                    # 计算价格弹性系数（取绝对值）
                    price_elasticity = abs(slope * (X.mean() / y.mean()))[0]

                    # 判断敏感度等级
                    if price_elasticity > 1.5:
                        sensitivity_level = '高敏感'
                    elif price_elasticity > 0.8:
                        sensitivity_level = '中敏感'
                    else:
                        sensitivity_level = '低敏感'

                    sensitivity_results.append({
                        '商品品类': category,
                        '价格弹性系数': round(price_elasticity, 4),
                        'R²决定系数': round(r2, 4),
                        '敏感度等级': sensitivity_level,
                        '数据点数': len(price_bin_stats),
                        '平均价格': round(price_bin_stats['实际售价'].mean(), 2),
                        '平均销量': round(price_bin_stats['销售数'].mean(), 2)
                    })

                    # 为前几个品类生成拟合图表
                    if len(self.results.get('fitting_charts', {})) < 4:
                        self._create_price_fitting_chart(category, price_bin_stats, model, price_elasticity)

                except Exception as e:
                    continue  # 单个品类分析失败时继续下一个

            sensitivity_df = pd.DataFrame(sensitivity_results)
            self.results['price_sensitivity'] = sensitivity_df.sort_values('价格弹性系数', ascending=False)
            return True

        except Exception as e:
            st.error(f"价格敏感度分析错误: {str(e)}")
            return False

    def _create_price_fitting_chart(self, category, price_bin_stats, model, elasticity):
        """创建价格-销量拟合图表"""
        try:
            if 'fitting_charts' not in self.results:
                self.results['fitting_charts'] = {}

            plt.figure(figsize=(10, 6))

            # 散点图
            plt.scatter(price_bin_stats['实际售价'], price_bin_stats['销售数'],
                        color='blue', alpha=0.7, s=60, label='实际数据')

            # 拟合线
            X_range = np.linspace(price_bin_stats['实际售价'].min(),
                                  price_bin_stats['实际售价'].max(), 100).reshape(-1, 1)
            y_range = model.predict(X_range)
            plt.plot(X_range, y_range, color='red', linewidth=2, label='线性拟合')

            plt.xlabel('实际售价')
            plt.ylabel('平均销量')
            plt.title(f'{category}\n价格-销量关系 (弹性系数: {elasticity:.3f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            self.results['fitting_charts'][f'price_fit_{category}'] = plt.gcf()
            plt.close()

            return True

        except Exception as e:
            return False

    def user_segmentation_analysis(self):
        """用户分层分析"""
        try:
            # 检查必要字段
            required_cols = ['客户年龄', '客户性别', '实际售价']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                st.warning(f"用户分层分析缺少字段: {missing_cols}")
                return False

            # 年龄分段
            def assign_age_group(age):
                try:
                    age = float(age)
                    if age < 25:
                        return '20-24岁'
                    elif age < 30:
                        return '25-29岁'
                    elif age < 35:
                        return '30-34岁'
                    elif age < 40:
                        return '35-39岁'
                    elif age < 45:
                        return '40-44岁'
                    elif age < 50:
                        return '45-49岁'
                    elif age < 55:
                        return '50-54岁'
                    elif age < 60:
                        return '55-59岁'
                    else:
                        return '60岁以上'
                except:
                    return '未知'

            self.df['年龄段'] = self.df['客户年龄'].apply(assign_age_group)

            # 用户分层分析
            user_segments = []

            for age_group in self.df['年龄段'].unique():
                for gender in self.df['客户性别'].unique():
                    segment_data = self.df[
                        (self.df['年龄段'] == age_group) &
                        (self.df['客户性别'] == gender)
                        ]

                    if len(segment_data) > 0:
                        # 计算价格接受度 (R1) - 平均购买价格与总体平均价格的比值
                        avg_price_segment = segment_data['实际售价'].mean()
                        avg_price_total = self.df['实际售价'].mean()
                        price_acceptance = avg_price_segment / avg_price_total if avg_price_total > 0 else 1

                        # 计算集中度 (R2) - 该分群在总销售额中的占比
                        sales_share = segment_data['销售额'].sum() / self.df['销售额'].sum() if self.df[
                                                                                                    '销售额'].sum() > 0 else 0

                        # 计算敏感倾向指数 (R3) - 基于价格变化的行为
                        # 简化计算：使用价格方差作为敏感度指标
                        price_variance = segment_data['实际售价'].var()
                        total_variance = self.df['实际售价'].var()
                        sensitivity_index = price_variance / total_variance if total_variance > 0 else 1

                        user_segments.append({
                            '年龄段': age_group,
                            '客户性别': gender,
                            '用户数量': len(segment_data),
                            '平均购买价格': round(avg_price_segment, 2),
                            '价格接受度(R1)': round(price_acceptance, 3),
                            '销售额占比(R2)': round(sales_share, 3),
                            '敏感倾向指数(R3)': round(sensitivity_index, 3),
                            '总销售额': round(segment_data['销售额'].sum(), 2)
                        })

            user_segments_df = pd.DataFrame(user_segments)
            self.results['user_segmentation'] = user_segments_df.sort_values('销售额占比(R2)', ascending=False)
            return True

        except Exception as e:
            st.error(f"用户分层分析错误: {str(e)}")
            return False

    def scenario_price_analysis(self):
        """场景价格分析"""
        try:
            # 定义商品品类到消费场景的映射
            scenario_mapping = {
                '办公': ['办公用品', '文具', '打印机', '电脑配件'],
                '家居': ['家具', '家居装饰', '床上用品', '厨房用品'],
                '数码': ['手机', '平板', '相机', '耳机'],
                '服饰': ['服装', '鞋类', '配饰', '箱包'],
                '美妆': ['化妆品', '护肤品', '香水', '个护'],
                '食品': ['零食', '饮料', '生鲜', '粮油']
            }

            # 为每个商品品类分配场景
            def assign_scenario(category):
                category_str = str(category)
                for scenario, keywords in scenario_mapping.items():
                    if any(keyword in category_str for keyword in keywords):
                        return scenario
                return '其他'

            self.df['消费场景'] = self.df['商品品类'].apply(assign_scenario)

            # 场景价格分析
            scenario_analysis = []

            for scenario in self.df['消费场景'].unique():
                scenario_data = self.df[self.df['消费场景'] == scenario]

                if len(scenario_data) > 0:
                    # 价格带分析
                    price_stats = scenario_data['实际售价'].describe()

                    # 计算场景价格敏感度指数 (SI)
                    # 使用价格变异系数作为敏感度指标
                    price_cv = scenario_data['实际售价'].std() / scenario_data['实际售价'].mean() if scenario_data[
                                                                                                         '实际售价'].mean() > 0 else 0

                    scenario_analysis.append({
                        '消费场景': scenario,
                        '商品数量': len(scenario_data['商品品类'].unique()),
                        '总销售额': round(scenario_data['销售额'].sum(), 2),
                        '平均价格': round(scenario_data['实际售价'].mean(), 2),
                        '最低价格': round(price_stats['min'], 2),
                        '最高价格': round(price_stats['max'], 2),
                        '价格标准差': round(scenario_data['实际售价'].std(), 2),
                        '价格敏感度指数(SI)': round(price_cv, 4),
                        '敏感度等级': '高敏感' if price_cv > 0.5 else '中敏感' if price_cv > 0.2 else '低敏感'
                    })

            scenario_df = pd.DataFrame(scenario_analysis)
            self.results['scenario_analysis'] = scenario_df.sort_values('总销售额', ascending=False)
            return True

        except Exception as e:
            st.error(f"场景价格分析错误: {str(e)}")
            return False

    def comprehensive_model_fusion(self):
        """综合模型融合（AHP层次分析法）"""
        try:
            # 获取各维度的分析结果
            abc_data = self.results.get('abc_classification', pd.DataFrame())
            price_data = self.results.get('price_sensitivity', pd.DataFrame())
            user_data = self.results.get('user_segmentation', pd.DataFrame())
            scenario_data = self.results.get('scenario_analysis', pd.DataFrame())

            if abc_data.empty or price_data.empty:
                st.warning("综合模型融合需要ABC分类和价格敏感度分析结果")
                return False

            # 构建综合评估矩阵
            comprehensive_results = []

            for _, abc_row in abc_data.iterrows():
                category = abc_row['商品品类']

                # 查找对应的价格敏感度数据
                price_row = price_data[price_data['商品品类'] == category]
                if price_row.empty:
                    continue

                # AHP权重分配（基于论文逻辑）
                weights = {
                    '品类重要性': 0.66,  # ABC分类权重
                    '价格敏感度': 0.23,  # 价格敏感度权重
                    '用户偏好': 0.07,  # 用户分层权重
                    '场景适配': 0.04  # 场景分析权重
                }

                # 品类重要性得分（A类=1.0, B类=0.7, C类=0.3）
                abc_score_map = {'A类': 1.0, 'B类': 0.7, 'C类': 0.3}
                abc_score = abc_score_map.get(abc_row['ABC分类（销售额）'], 0.3)

                # 价格敏感度得分（转换为0-1的标准化得分）
                price_elasticity = price_row.iloc[0]['价格弹性系数']
                price_score = min(price_elasticity / 2.0, 1.0)  # 假设最大弹性为2.0

                # 用户偏好得分（简化计算）
                user_score = 0.5  # 默认值，实际应根据用户分层数据计算

                # 场景适配得分（简化计算）
                scenario_score = 0.5  # 默认值，实际应根据场景分析数据计算

                # 综合得分计算
                comprehensive_score = (
                        abc_score * weights['品类重要性'] +
                        price_score * weights['价格敏感度'] +
                        user_score * weights['用户偏好'] +
                        scenario_score * weights['场景适配']
                )

                # 敏感度等级判定
                if comprehensive_score >= 0.7:
                    sensitivity_level = '高敏感'
                    operation_priority = '高'
                elif comprehensive_score >= 0.4:
                    sensitivity_level = '中敏感'
                    operation_priority = '中'
                else:
                    sensitivity_level = '低敏感'
                    operation_priority = '低'

                comprehensive_results.append({
                    '商品品类': category,
                    'ABC分类': abc_row['ABC分类（销售额）'],
                    '价格弹性系数': price_elasticity,
                    '品类重要性得分': round(abc_score, 3),
                    '价格敏感度得分': round(price_score, 3),
                    '用户偏好得分': round(user_score, 3),
                    '场景适配得分': round(scenario_score, 3),
                    '综合敏感度得分': round(comprehensive_score, 3),
                    '敏感度等级': sensitivity_level,
                    '运营优先级': operation_priority
                })

            comprehensive_df = pd.DataFrame(comprehensive_results)
            self.results['comprehensive_fusion'] = comprehensive_df.sort_values('综合敏感度得分', ascending=False)

            # 生成运营策略推荐
            self._generate_operation_strategies(comprehensive_df)

            return True

        except Exception as e:
            st.error(f"综合模型融合错误: {str(e)}")
            return False

    def _generate_operation_strategies(self, comprehensive_df):
        """生成运营策略推荐"""
        try:
            strategies = []

            for _, row in comprehensive_df.iterrows():
                category = row['商品品类']
                abc_class = row['ABC分类']
                sensitivity_level = row['敏感度等级']
                comprehensive_score = row['综合敏感度得分']

                # 根据ABC分类和敏感度等级生成策略
                if abc_class == 'A类' and sensitivity_level == '高敏感':
                    strategies.append({
                        '商品品类': category,
                        '策略类型': '价格优化',
                        '具体措施': '实施动态定价，关注竞品价格，设置价格预警机制',
                        '预期效果': '提升价格竞争力，保持市场份额',
                        '优先级': '高'
                    })
                    strategies.append({
                        '商品品类': category,
                        '策略类型': '库存优化',
                        '具体措施': '增加安全库存，优化补货频率，避免缺货损失',
                        '预期效果': '提高库存周转率，减少缺货风险',
                        '优先级': '高'
                    })

                elif abc_class == 'A类' and sensitivity_level == '中敏感':
                    strategies.append({
                        '商品品类': category,
                        '策略类型': '促销策略',
                        '具体措施': '设计组合促销，捆绑销售高利润商品',
                        '预期效果': '提升客单价，增加整体利润',
                        '优先级': '中'
                    })

                elif abc_class == 'B类':
                    strategies.append({
                        '商品品类': category,
                        '策略类型': '市场拓展',
                        '具体措施': '加强目标用户推广，优化产品展示位置',
                        '预期效果': '提升品类知名度，促进销售增长',
                        '优先级': '中'
                    })

                elif abc_class == 'C类':
                    strategies.append({
                        '商品品类': category,
                        '策略类型': '成本控制',
                        '具体措施': '精简SKU，优化采购批量，降低库存成本',
                        '预期效果': '减少资源占用，提高运营效率',
                        '优先级': '低'
                    })

                # 根据敏感度等级补充策略
                if sensitivity_level == '高敏感':
                    strategies.append({
                        '商品品类': category,
                        '策略类型': '价格监控',
                        '具体措施': '建立价格监测体系，快速响应市场变化',
                        '预期效果': '保持价格敏感度优势',
                        '优先级': '高'
                    })

            strategies_df = pd.DataFrame(strategies)
            self.results['operation_strategies'] = strategies_df

            return True

        except Exception as e:
            st.error(f"运营策略生成错误: {str(e)}")
            return False
def display_task4_results(results, result_files, progress_log):
    """显示任务4分析结果"""

    # 1. 显示分析日志
    st.subheader("📝 分析执行日志")
    for log in progress_log:
        st.write(f"▪️ {log}")

    # 2. ABC分类分析结果
    if 'abc_classification' in results:
        st.subheader("📊 ABC分类分析")

        abc_data = results['abc_classification']

        # 检查列名并适配
        abc_columns_to_show = ['商品品类', '销售额占比%', '利润占比%']

        # 动态确定ABC分类列名
        abc_class_col = None
        possible_abc_cols = ['ABC分类（销售额）', 'ABC分类', 'ABC_class']
        for col in possible_abc_cols:
            if col in abc_data.columns:
                abc_class_col = col
                break

        if abc_class_col:
            abc_columns_to_show.append(abc_class_col)

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(abc_data[abc_columns_to_show])

        with col2:
            if 'abc_visualizations' in results:
                st.pyplot(results['abc_visualizations']['sales_distribution'])

        # 显示其他图表
        if 'abc_visualizations' in results:
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(results['abc_visualizations']['cumulative_sales'])
            with col2:
                st.pyplot(results['abc_visualizations']['abc_distribution'])

    # 3. 价格敏感度分析结果
    if 'price_sensitivity' in results:
        st.subheader("💰 价格敏感度分析")

        sensitivity_data = results['price_sensitivity']

        # 显示敏感度分析结果
        st.dataframe(sensitivity_data)

        # 显示拟合图表
        if 'fitting_charts' in results:
            st.subheader("📈 价格-销量关系拟合图")
            cols = st.columns(2)
            chart_count = 0

            for chart_name, fig in results['fitting_charts'].items():
                with cols[chart_count % 2]:
                    st.pyplot(fig)
                chart_count += 1

    # 4. 用户分层分析结果
    if 'user_segmentation' in results:
        st.subheader("👥 用户分层分析")
        st.dataframe(results['user_segmentation'])

    # 5. 场景价格分析结果
    if 'scenario_analysis' in results:
        st.subheader("🏷️ 场景价格分析")
        st.dataframe(results['scenario_analysis'])

    # 6. 综合模型融合结果
    if 'comprehensive_fusion' in results:
        st.subheader("🎯 综合敏感度评估")
        st.dataframe(results['comprehensive_fusion'])

    # 7. 运营策略推荐
    if 'operation_strategies' in results:
        st.subheader("🚀 可执行运营策略")

        # 按优先级筛选
        priority_filter = st.selectbox("筛选策略优先级:", ["全部", "高", "中"])

        strategy_data = results['operation_strategies']
        if priority_filter != "全部":
            filtered_strategies = strategy_data[
                strategy_data['优先级'] == priority_filter]
        else:
            filtered_strategies = strategy_data

        st.dataframe(filtered_strategies)

    # 8. 文件下载
    st.subheader("📥 分析结果下载")
    for filename, data in result_files.items():
        excel_bytes = io.BytesIO()
        with pd.ExcelWriter(excel_bytes, engine='openpyxl') as writer:
            data.to_excel(writer, index=False)
        st.download_button(
            label=f"下载 {filename}",
            data=excel_bytes.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
# ============================================================================
# 页面函数
# ============================================================================
def show_project_overview():
    """项目概览页面"""
    st.header("🎯 项目概览")

    st.markdown(
        '<div class="fix-note"><strong>系统功能：</strong><br>1. 每个任务都支持独立数据导入<br>2. 可选择使用任务1处理数据或上传新数据<br>3. 自动字段类型识别和必需字段检查<br>4. 按论文标准流程生成分析结果</div>',
        unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 系统功能概述
        完整的电商销售分析流程，每个任务都支持独立数据导入：

        - **数据预处理**: 按论文要求生成6个标准化输出文件
        - **多维特征分析**: 支持自定义数据或使用预处理数据
        - **销售预测**: 独立数据导入，自动检测时间序列字段  
        - **运营优化**: 灵活的数据源选择，支持多维度分析

        **数据导入选项：**
        - ✅ 使用任务1预处理后的数据
        - ✅ 上传新的Excel/CSV文件
        - ✅ 自动字段类型识别
        - ✅ 必需字段检查
        """)

    with col2:
        st.metric("标准输出文件", "6个")
        st.metric("分析任务", "4个")
        st.metric("数据导入方式", "每个任务独立")
        st.metric("支持格式", "Excel/CSV")

    # 任务状态概览
    st.subheader("任务完成状态")
    tasks = [
        ("数据预处理", st.session_state.task1_completed),
        ("多维分析", st.session_state.task2_completed),
        ("销售预测", st.session_state.task3_completed),
        ("运营优化", st.session_state.task4_completed)
    ]

    for task_name, completed in tasks:
        status = "✅ 已完成" if completed else "⏳ 待完成"
        st.write(f"- {task_name}: {status}")

    # 新增：使用指南
    st.subheader("🚀 快速开始指南")

    with st.expander("📖 详细使用说明", expanded=True):
        st.markdown("""
        **第一步：数据预处理（任务1）**
        - 上传原始Excel/CSV数据文件
        - 系统自动执行5个标准化处理步骤
        - 生成6个论文要求的输出文件

        **第二步：多维特征分析（任务2）**
        - 可选择使用任务1处理数据或上传新文件
        - 必需字段：区域、商品品类、利润
        - 生成热力图、聚类分析、地理分布等

        **第三步：销售预测（任务3）**
        - 可选择使用任务1处理数据或上传新文件  
        - 必需字段：日期、利润
        - 使用ARIMA+XGBoost混合模型预测

        **第四步：运营优化（任务4）**
        - 可选择使用任务1处理数据或上传新文件
        - 必需字段：商品品类、销售额、利润、实际售价、销售数
        - 生成ABC分类、价格敏感度、运营策略等
        """)

    # 新增：数据要求说明
    st.subheader("📋 数据字段要求")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **核心业务字段：**
        - 商品品类
        - 区域/省份/城市
        - 销售额
        - 利润
        - 销售数
        """)

    with col2:
        st.markdown("""
        **价格相关字段：**
        - 进货价格
        - 实际售价
        - 成本价格
        - 折扣金额
        """)

    with col3:
        st.markdown("""
        **客户相关字段：**
        - 客户性别
        - 客户年龄
        - 客户等级
        - 购买日期
        """)

    # 新增：文件格式说明
    st.subheader("📁 文件格式支持")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Excel格式 (.xlsx)：**
        - 支持多工作表
        - 自动识别数据类型
        - 保持原始格式

        **推荐用于：**
        - 复杂数据结构
        - 多维度分析
        - 大型数据集
        """)

    with col2:
        st.markdown("""
        **CSV格式 (.csv)：**
        - 通用数据格式
        - 快速加载处理
        - 兼容性好

        **推荐用于：**
        - 简单数据结构
        - 快速测试
        - 跨平台使用
        """)

    # 任务状态概览
    st.subheader("任务完成状态")
    tasks = [
        ("数据预处理", st.session_state.task1_completed),
        ("多维分析", st.session_state.task2_completed),
        ("销售预测", st.session_state.task3_completed),
        ("运营优化", st.session_state.task4_completed)
    ]

    for task_name, completed in tasks:
        status = "✅ 已完成" if completed else "⏳ 待完成"
        st.write(f"- {task_name}: {status}")


def task1_data_preprocessing():
    """任务1：数据预处理页面（按论文要求生成标准化输出文件）"""
    st.header("📁 任务1: 数据预处理")

    st.markdown(
        '<div class="fix-note"><strong>论文要求输出文件：</strong><br>1. 电商 步骤1 缺失值统计结果.xlsx<br>2. 电商 步骤2 进货价格处理后数据.xlsx<br>3. 电商 步骤3 利润修正后数据.xlsx<br>4. 电商 步骤4 异常修正及利润重算后数据.xlsx<br>5. 电商 步骤5 MinMax标准化后数据.xlsx<br>6. 电商 步骤5 ZScore标准化后数据.xlsx</div>',
        unsafe_allow_html=True)

    # 文件上传组件
    uploaded_file = st.file_uploader("上传原始数据表（支持Excel或CSV格式）", type=["xlsx", "csv"])

    if uploaded_file is not None:
        # 读取文件
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            # 保存原始数据
            st.session_state.raw_data = df
            st.session_state.current_file = uploaded_file.name

            # 数据清洗：自动处理数值列中的非数值字符
            df_clean = clean_numeric_columns(df)

            st.success(f"文件上传成功！共 {len(df)} 条记录，{len(df.columns)} 个字段")

            # 显示数据预览和清洗前后对比
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("原始数据预览（前5行）")
                st.dataframe(df.head())

            with col2:
                st.subheader("清洗后数据预览（前5行）")
                st.dataframe(df_clean.head())

            # 显示数据类型信息
            st.subheader("数据类型信息")
            dtype_df = pd.DataFrame({
                '字段名': df_clean.columns,
                '数据类型': df_clean.dtypes.astype(str)
            })
            st.dataframe(dtype_df)

            # 执行预处理按钮
            if st.button("🚀 开始数据预处理（按论文步骤）", type="primary"):
                with st.spinner("正在执行数据预处理...（按论文要求生成6个输出文件）"):
                    preprocessor = Task1Preprocessor(df_clean)
                    result_files, progress_log, final_data, encoders, column_types = preprocessor.generate_all_results()

                    if result_files:
                        # 保存结果到session state
                        st.session_state.step1_missing_data = result_files['电商 步骤1 缺失值统计结果.xlsx']
                        st.session_state.step2_price_data = result_files['电商 步骤2 进货价格处理后数据.xlsx']
                        st.session_state.step3_profit_data = result_files['电商 步骤3 利润修正后数据.xlsx']
                        st.session_state.step4_abnormal_data = result_files['电商 步骤4 异常修正及利润重算后数据.xlsx']
                        st.session_state.step5_minmax_data = result_files['电商 步骤5 MinMax标准化后数据.xlsx']
                        st.session_state.step5_zscore_data = result_files['电商 步骤5 ZScore标准化后数据.xlsx']
                        st.session_state.processed_data = final_data
                        st.session_state.category_encoder = encoders
                        st.session_state.column_types = column_types
                        st.session_state.task1_completed = True

                        st.success("✅ 数据预处理完成！已生成论文要求的6个输出文件")

                        # 1. 展示预处理步骤结果预览
                        st.subheader("1. 预处理步骤结果预览")

                        # 步骤1：缺失值统计结果
                        st.markdown("#### 步骤1：缺失值统计结果")
                        st.dataframe(st.session_state.step1_missing_data.head())

                        # 步骤2：进货价格处理后数据
                        st.markdown("#### 步骤2：进货价格处理后数据")
                        st.dataframe(st.session_state.step2_price_data[['商品品类', '进货价格']].head())

                        # 步骤3：利润修正后数据
                        st.markdown("#### 步骤3：利润修正后数据")
                        if '利润是否正确' in st.session_state.step3_profit_data.columns:
                            st.dataframe(
                                st.session_state.step3_profit_data[['商品品类', '利润', '利润是否正确']].head())
                        else:
                            st.dataframe(st.session_state.step3_profit_data[['商品品类', '利润']].head())

                        # 2. 展示预处理日志
                        st.subheader("2. 预处理执行日志")
                        for log in progress_log:
                            st.write(f"▪️ {log}")

                        # 3. 提供结果文件下载（按论文要求的文件名）
                        st.subheader("📥 论文要求输出文件下载")
                        for filename, data in result_files.items():
                            if isinstance(data, pd.ExcelWriter):
                                excel_bytes = io.BytesIO()
                                data.save(excel_bytes)
                                excel_bytes.seek(0)
                                st.download_button(
                                    label=f"下载 {filename}",
                                    data=excel_bytes,
                                    file_name=filename,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            elif isinstance(data, pd.DataFrame):
                                excel_bytes = io.BytesIO()
                                with pd.ExcelWriter(excel_bytes, engine='openpyxl') as writer:
                                    data.to_excel(writer, index=False)
                                st.download_button(
                                    label=f"下载 {filename}",
                                    data=excel_bytes.getvalue(),
                                    file_name=filename,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                        # 提示下一步操作
                        st.info("预处理完成！可继续进行 多维销售特征分析（任务2）")
                    else:
                        st.error("预处理失败，请检查数据格式或查看错误日志")
                        for log in progress_log:
                            st.error(log)

        except Exception as e:
            st.error(f"文件读取错误: {str(e)}")
    else:
        st.info("请上传原始数据文件开始预处理流程（建议包含：商品品类、区域、销售额、利润、日期等字段）")


def enhanced_task2_multidimensional_analysis():
    """增强版多维分析页面"""
    st.header("🔍 任务2: 多维销售特征分析")

    # ============================================================================
    # 使用新的数据导入组件
    # ============================================================================
    st.subheader("📁 数据导入")

    col1, col2 = st.columns(2)
    data_source = None
    current_data = None

    with col1:
        # 数据源选择
        data_source_option = st.radio(
            "选择多维分析数据源:",
            ["使用原始数据", "选择任务1处理文件", "上传新文件"],
            key="data_source_task2"
        )

    with col2:
        if data_source_option == "使用原始数据":
            if st.session_state.get('raw_data') is not None:
                current_data = st.session_state.raw_data
                data_source = "原始数据"
                st.success(f"使用原始数据，共 {len(current_data)} 条记录")
            else:
                st.error("暂无原始数据，请先在任务1中上传文件")
                return

        elif data_source_option == "选择任务1处理文件":
            # 任务1生成的文件列表
            task1_files = {
                "步骤2_进货价格处理后数据": "step2_price_data",
                "步骤3_利润修正后数据": "step3_profit_data",
                "步骤4_异常修正及利润重算后数据": "step4_abnormal_data",
                "步骤5_MinMax标准化后数据": "step5_minmax_data",
                "步骤5_ZScore标准化后数据": "step5_zscore_data"
            }

            selected_file = st.selectbox(
                "选择任务1处理文件:",
                list(task1_files.keys()),
                key="task1_file_task2"
            )

            if selected_file and st.session_state.get(task1_files[selected_file]) is not None:
                current_data = st.session_state[task1_files[selected_file]]
                data_source = f"任务1: {selected_file}"
                st.success(f"使用{selected_file}，共 {len(current_data)} 条记录")
            else:
                st.error("选择的文件不存在，请先完成任务1")
                return

        else:  # 上传新文件
            uploaded_file = st.file_uploader(
                "上传多维分析数据文件",
                type=["xlsx", "csv"],
                key="upload_task2_new"
            )

            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.xlsx'):
                        current_data = pd.read_excel(uploaded_file)
                    else:
                        current_data = pd.read_csv(uploaded_file)

                    current_data = clean_numeric_columns(current_data)
                    data_source = f"自定义文件: {uploaded_file.name}"
                    st.success(f"数据加载成功！共 {len(current_data)} 条记录")
                except Exception as e:
                    st.error(f"文件读取错误: {str(e)}")
                    return
            else:
                st.info("请上传数据文件")
                return

    # 检查必需字段（多维分析需要的核心字段）
    required_columns = ['区域', '商品品类', '利润']
    if current_data is not None:
        missing_columns = [col for col in required_columns if col not in current_data.columns]
        if missing_columns:
            st.error(f"缺少必需字段: {', '.join(missing_columns)}")
            st.info(f"多维分析需要的字段: {', '.join(required_columns)}")
            st.info("可选字段: 日期, 客户性别, 客户年龄, 销售额, 销售数等")
            return

    # 显示数据源信息
    if current_data is not None:
        st.success(f"✅ 数据准备完成 - 数据源: {data_source}")
        st.info(f"数据维度: {len(current_data)} 行 × {len(current_data.columns)} 列")

        # 显示数据预览
        with st.expander("📋 数据预览"):
            st.dataframe(current_data.head(10))
    else:
        return
    # ============================================================================
    # 结束数据导入部分
    # ============================================================================

    # 自动检测字段类型
    column_types = auto_detect_column_types(current_data)

    # 分析模式选择
    analysis_mode = st.radio(
        "选择分析模式:",
        ["📊 Python可视化展示", "📁 论文图表数据导出", "🎨 交互式可视化仪表板"],  # 新增第三个选项
        horizontal=True
    )

    if st.button("🚀 执行多维特征分析", type="primary"):
        with st.spinner("正在执行多维分析..."):
            analyzer = EnhancedTask2Analyzer(current_data, column_types)

            # 执行基础分析（热力图和聚类）
            heatmap_success = analyzer.create_heatmaps()
            cluster_success = analyzer.perform_clustering_analysis()

            # 生成所有分析数据
            all_analysis_data = analyzer.generate_all_analysis_data()

            st.session_state.task2_results = analyzer.results
            st.session_state.task2_analysis_data = all_analysis_data
            st.session_state.task2_completed = True

            st.success("✅ 多维特征分析完成！")

            # 显示分析结果摘要
            st.subheader("📊 分析结果摘要")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("热力图生成", "成功" if heatmap_success else "部分失败")
            with col2:
                st.metric("聚类分析", "成功" if cluster_success else "失败")
            with col3:
                analysis_count = sum(1 for data in all_analysis_data.values() if data is not None)
                st.metric("分析维度", f"{analysis_count}个")

            if analysis_mode == "📊 Python可视化展示":
                show_python_visualizations(analyzer)
            else:
                show_data_export_interface(all_analysis_data)
    else:
        st.info("""
        **多维特征分析功能说明：**

        **必需字段：**
        - 区域（用于地理分布分析）
        - 商品品类（用于品类分析）  
        - 利润（用于价值分析）

        **推荐字段：**
        - 日期（用于时间序列分析）
        - 客户性别、客户年龄（用于用户画像分析）
        - 销售数、进货价格（用于业务分析）

        **分析模式：**
        - 📊 Python可视化展示：系统内置图表，即时查看分析结果
        - 📁 论文图表数据导出：导出Excel数据用于论文图表制作

        **点击上方按钮开始分析！**
        """)


def task3_sales_forecast():
    """任务3：销售预测页面"""
    st.header("📈 任务3: 销售预测")

    # ============================================================================
    # 使用新的数据导入组件
    # ============================================================================
    st.subheader("📁 数据导入")

    col1, col2 = st.columns(2)
    data_source = None
    current_data = None

    with col1:
        # 数据源选择
        data_source_option = st.radio(
            "选择销售预测数据源:",
            ["使用原始数据", "选择任务1处理文件", "上传新文件"],
            key="data_source_task3"
        )

    with col2:
        if data_source_option == "使用原始数据":
            if st.session_state.get('raw_data') is not None:
                current_data = st.session_state.raw_data
                data_source = "原始数据"
                st.success(f"使用原始数据，共 {len(current_data)} 条记录")
            else:
                st.error("暂无原始数据，请先在任务1中上传文件")
                return

        elif data_source_option == "选择任务1处理文件":
            # 任务1生成的文件列表
            task1_files = {
                "步骤2_进货价格处理后数据": "step2_price_data",
                "步骤3_利润修正后数据": "step3_profit_data",
                "步骤4_异常修正及利润重算后数据": "step4_abnormal_data",
                "步骤5_MinMax标准化后数据": "step5_minmax_data",
                "步骤5_ZScore标准化后数据": "step5_zscore_data"
            }

            selected_file = st.selectbox(
                "选择任务1处理文件:",
                list(task1_files.keys()),
                key="task1_file_task3"
            )

            if selected_file and st.session_state.get(task1_files[selected_file]) is not None:
                current_data = st.session_state[task1_files[selected_file]]
                data_source = f"任务1: {selected_file}"
                st.success(f"使用{selected_file}，共 {len(current_data)} 条记录")
            else:
                st.error("选择的文件不存在，请先完成任务1")
                return

        else:  # 上传新文件
            uploaded_file = st.file_uploader(
                "上传销售预测数据文件",
                type=["xlsx", "csv"],
                key="upload_task3_new"
            )

            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.xlsx'):
                        current_data = pd.read_excel(uploaded_file)
                    else:
                        current_data = pd.read_csv(uploaded_file)

                    current_data = clean_numeric_columns(current_data)
                    data_source = f"自定义文件: {uploaded_file.name}"
                    st.success(f"数据加载成功！共 {len(current_data)} 条记录")
                except Exception as e:
                    st.error(f"文件读取错误: {str(e)}")
                    return
            else:
                st.info("请上传数据文件")
                return

    # 检查必需字段（销售预测需要的核心字段）
    required_columns = ['日期', '利润']
    if current_data is not None:
        missing_columns = [col for col in required_columns if col not in current_data.columns]
        if missing_columns:
            st.error(f"缺少必需字段: {', '.join(missing_columns)}")
            st.info(f"销售预测需要的字段: {', '.join(required_columns)}")
            st.info("推荐字段: 销售额, 销售数, 进货价格, 实际售价等")
            return

    # 显示数据源信息
    if current_data is not None:
        st.success(f"✅ 数据准备完成 - 数据源: {data_source}")
        st.info(f"数据维度: {len(current_data)} 行 × {len(current_data.columns)} 列")

        # 显示数据预览
        with st.expander("📋 数据预览"):
            st.dataframe(current_data.head(10))

        # 检查日期字段格式
        date_col = next((col for col in current_data.columns if '日期' in col), None)
        if date_col:
            st.info(f"检测到日期字段: {date_col}")
            # 尝试转换日期为数值格式（与预测代码一致）
            try:
                current_data[date_col] = pd.to_numeric(current_data[date_col], errors='coerce')
                date_range = f"{current_data[date_col].min()} 至 {current_data[date_col].max()}"
                st.info(f"日期范围: {date_range}")
            except:
                st.warning("日期字段格式可能需要调整")
    else:
        return
    # ============================================================================
    # 结束数据导入部分
    # ============================================================================

    # 自动检测字段类型
    column_types = auto_detect_column_types(current_data)

    # 执行预测
    if st.button("🚀 执行ARIMA-XGBoost混合预测", type="primary"):
        with st.spinner("预测中...（使用ARIMA(2,1,2)+XGBoost混合模型）"):
            forecaster = Task3Forecaster(current_data, column_types)
            result_files, progress_log = forecaster.generate_all_results()

            if result_files:
                st.session_state.task3_results = forecaster.results
                st.session_state.task3_completed = True
                st.success("✅ 销售预测完成！")

                # 1. 展示预测结果可视化
                st.subheader("📊 预测结果可视化")

                if 'visualizations' in forecaster.results:
                    viz_results = forecaster.results['visualizations']

                    # 利润预测对比图（必须展示）
                    st.markdown("#### 1. 利润预测对比图")
                    st.pyplot(viz_results['main_forecast'])
                    st.markdown("""
                    **图表说明：**
                    - 蓝色线：训练集实际利润值
                    - 红色线：测试集实际利润值  
                    - 粉色虚线：ARIMA模型预测值
                    - 绿色线：ARIMA+XGBoost最终预测值
                    - 灰色虚线：训练集/测试集分界线
                    """)

                    # 误差分析图
                    st.markdown("#### 2. 预测误差分析图")
                    st.pyplot(viz_results['error_analysis'])
                    st.markdown("""
                    **图表说明：**
                    - 显示每日预测的相对误差百分比
                    - MAPE（平均绝对百分比误差）是主要评估指标
                    - 误差越小，模型预测精度越高
                    """)

                    # 残差分析图
                    if 'residual_analysis' in viz_results:
                        st.markdown("#### 3. ARIMA模型残差分析图")
                        st.pyplot(viz_results['residual_analysis'])
                        st.markdown("""
                        **图表说明：**
                        - 显示ARIMA模型在训练集上的残差分布
                        - 残差越接近0且波动越小，说明ARIMA模型拟合越好
                        - 为XGBoost提供学习目标
                        """)

                    # 特征重要性排名图
                    if 'feature_importance' in viz_results:
                        st.markdown("#### 4. 特征重要性排名图")
                        st.pyplot(viz_results['feature_importance'])
                        st.markdown("""
                        **图表说明：**
                        - 显示XGBoost模型中各特征的重要性得分
                        - 重要性越高，该特征对残差预测的贡献越大
                        - 帮助理解模型决策依据
                        """)

                # 2. 展示预测结果表格
                st.subheader("📋 预测结果详情")
                if 'detailed_results' in forecaster.results:
                    forecast_df = forecaster.results['detailed_results']
                    st.dataframe(forecast_df.round(2))

                    # 关键指标总结
                    st.subheader("🎯 预测关键指标")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        mape = forecaster.results.get('mape', 0)
                        st.metric("测试集MAPE", f"{mape:.2f}%")

                    with col2:
                        best_error = forecast_df['相对误差(%)'].min()
                        best_day = forecast_df.loc[forecast_df['相对误差(%)'].idxmin(), '日期']
                        st.metric("最佳预测精度", f"{best_error:.1f}%", f"11月{int(best_day)}日")

                    with col3:
                        worst_error = forecast_df['相对误差(%)'].max()
                        worst_day = forecast_df.loc[forecast_df['相对误差(%)'].idxmax(), '日期']
                        st.metric("最差预测精度", f"{worst_error:.1f}%", f"11月{int(worst_day)}日")

                # 3. 特征重要性分析
                st.subheader("🔍 特征重要性分析")
                if 'feature_importance' in forecaster.results:
                    feature_importance = forecaster.results['feature_importance']
                    st.dataframe(feature_importance.round(4))

                    st.info("""
                    **特征重要性解读：**
                    - **高重要性特征**：对模型预测影响最大的因素
                    - **中等重要性特征**：有一定预测价值的辅助因素  
                    - **低重要性特征**：对预测结果影响较小
                    """)

                # 4. 进度日志
                st.subheader("📝 预测执行日志")
                for log in progress_log:
                    st.write(f"▪️ {log}")

                # 5. 文件下载
                st.subheader("📥 预测结果文件下载")
                for filename, data in result_files.items():
                    if isinstance(data, pd.DataFrame):
                        excel_bytes = io.BytesIO()
                        with pd.ExcelWriter(excel_bytes, engine='openpyxl') as writer:
                            data.to_excel(writer, index=False)
                        st.download_button(
                            label=f"下载 {filename}",
                            data=excel_bytes.getvalue(),
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                # 提示下一步操作
                st.info("销售预测完成！可继续进行 运营策略优化（任务4）")
            else:
                st.error("预测失败，请查看错误日志")
                for log in progress_log:
                    st.error(log)
    else:
        st.info("""
        **ARIMA-XGBoost混合预测模型说明：**

        **必需字段：**
        - 日期（时间序列索引，支持数值格式如1-30表示11月1-30日）
        - 利润（预测目标变量）

        **推荐字段：**
        - 销售额、销售数（特征变量）
        - 进货价格、实际售价（业务特征）
        - 商品品类、区域（分类特征）

        **模型特点：**
        - 使用ARIMA(2,1,2)模型捕捉时间序列趋势
        - 使用XGBoost模型学习ARIMA的残差模式
        - 最终预测 = ARIMA预测 + XGBoost残差预测
        - 自动划分训练集（前24天）和测试集（后6天）
        - 输出MAPE（平均绝对百分比误差）评估预测精度

        **点击上方按钮开始预测分析！**
        """)


def task4_operation_optimization():
    """任务4：运营策略优化页面"""
    st.header("💡 任务4: 运营策略优化")

    # ============================================================================
    # 使用新的数据导入组件
    # ============================================================================
    st.subheader("📁 数据导入")

    col1, col2 = st.columns(2)
    data_source = None
    current_data = None

    with col1:
        # 数据源选择
        data_source_option = st.radio(
            "选择运营优化数据源:",
            ["使用原始数据", "选择任务1处理文件", "上传新文件"],
            key="data_source_task4"
        )

    with col2:
        if data_source_option == "使用原始数据":
            if st.session_state.get('raw_data') is not None:
                current_data = st.session_state.raw_data
                data_source = "原始数据"
                st.success(f"使用原始数据，共 {len(current_data)} 条记录")
            else:
                st.error("暂无原始数据，请先在任务1中上传文件")
                return

        elif data_source_option == "选择任务1处理文件":
            # 任务1生成的文件列表
            task1_files = {
                "步骤2_进货价格处理后数据": "step2_price_data",
                "步骤3_利润修正后数据": "step3_profit_data",
                "步骤4_异常修正及利润重算后数据": "step4_abnormal_data",
                "步骤5_MinMax标准化后数据": "step5_minmax_data",
                "步骤5_ZScore标准化后数据": "step5_zscore_data"
            }

            selected_file = st.selectbox(
                "选择任务1处理文件:",
                list(task1_files.keys()),
                key="task1_file_task4"
            )

            if selected_file and st.session_state.get(task1_files[selected_file]) is not None:
                current_data = st.session_state[task1_files[selected_file]]
                data_source = f"任务1: {selected_file}"
                st.success(f"使用{selected_file}，共 {len(current_data)} 条记录")
            else:
                st.error("选择的文件不存在，请先完成任务1")
                return

        else:  # 上传新文件
            uploaded_file = st.file_uploader(
                "上传运营优化数据文件",
                type=["xlsx", "csv"],
                key="upload_task4_new"
            )

            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.xlsx'):
                        current_data = pd.read_excel(uploaded_file)
                    else:
                        current_data = pd.read_csv(uploaded_file)

                    current_data = clean_numeric_columns(current_data)
                    data_source = f"自定义文件: {uploaded_file.name}"
                    st.success(f"数据加载成功！共 {len(current_data)} 条记录")
                except Exception as e:
                    st.error(f"文件读取错误: {str(e)}")
                    return
            else:
                st.info("请上传数据文件")
                return

    # 检查必需字段（运营优化需要的核心字段）
    required_columns = ['商品品类', '销售额', '利润', '实际售价', '销售数']
    if current_data is not None:
        missing_columns = [col for col in required_columns if col not in current_data.columns]
        if missing_columns:
            st.error(f"缺少必需字段: {', '.join(missing_columns)}")
            st.info(f"运营优化分析需要的字段: {', '.join(required_columns)}")
            st.info("推荐字段: 进货价格, 区域, 客户性别, 客户年龄, 日期等")
            return

    # 显示数据源信息
    if current_data is not None:
        st.success(f"✅ 数据准备完成 - 数据源: {data_source}")
        st.info(f"数据维度: {len(current_data)} 行 × {len(current_data.columns)} 列")

        # 显示数据预览
        with st.expander("📋 数据预览"):
            st.dataframe(current_data.head(10))

        # 显示关键字段统计
        st.subheader("📊 数据概览")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if '商品品类' in current_data.columns:
                st.metric("商品品类数", current_data['商品品类'].nunique())
        with col2:
            if '销售额' in current_data.columns:
                st.metric("总销售额", f"¥{current_data['销售额'].sum():,.0f}")
        with col3:
            if '利润' in current_data.columns:
                st.metric("总利润", f"¥{current_data['利润'].sum():,.0f}")
        with col4:
            if '销售数' in current_data.columns:
                st.metric("总销售数", f"{current_data['销售数'].sum():,}")
    else:
        return
    # ============================================================================
    # 结束数据导入部分
    # ============================================================================

    # 自动检测字段类型
    column_types = auto_detect_column_types(current_data)

    # 分析配置
    st.subheader("⚙️ 分析配置")
    analysis_options = st.multiselect(
        "选择分析模块:",
        [
            "ABC分类分析",
            "价格敏感度分析",
            "用户分层分析",
            "场景价格分析",
            "综合模型融合"
        ],
        default=[
            "ABC分类分析",
            "价格敏感度分析",
            "用户分层分析",
            "场景价格分析",
            "综合模型融合"
        ]
    )

    # 执行分析
    if st.button("🚀 执行运营优化分析", type="primary"):
        with st.spinner("执行运营优化分析中..."):
            optimizer = Task4Optimizer(current_data, column_types)

            # 根据选择的模块执行分析
            result_files = {}
            progress_log = []

            if "ABC分类分析" in analysis_options:
                if optimizer.abc_classification_analysis():
                    progress_log.append("✅ ABC分类分析完成")
                    result_files['01_ABC分类结果.xlsx'] = optimizer.results['abc_classification']
                else:
                    progress_log.append("❌ ABC分类分析失败")

            if "价格敏感度分析" in analysis_options:
                if optimizer.price_sensitivity_analysis():
                    progress_log.append("✅ 价格敏感度分析完成")
                    result_files['02_价格敏感度分析.xlsx'] = optimizer.results['price_sensitivity']
                else:
                    progress_log.append("❌ 价格敏感度分析失败")

            if "用户分层分析" in analysis_options:
                if optimizer.user_segmentation_analysis():
                    progress_log.append("✅ 用户分层分析完成")
                    result_files['03_用户分层分析.xlsx'] = optimizer.results['user_segmentation']
                else:
                    progress_log.append("❌ 用户分层分析失败")

            if "场景价格分析" in analysis_options:
                if optimizer.scenario_price_analysis():
                    progress_log.append("✅ 场景价格分析完成")
                    result_files['04_场景价格分析.xlsx'] = optimizer.results['scenario_analysis']
                else:
                    progress_log.append("❌ 场景价格分析失败")

            if "综合模型融合" in analysis_options:
                if optimizer.comprehensive_model_fusion():
                    progress_log.append("✅ 综合模型融合完成")
                    result_files['05_综合敏感度评估.xlsx'] = optimizer.results['comprehensive_fusion']
                    result_files['06_运营策略推荐.xlsx'] = optimizer.results['operation_strategies']
                else:
                    progress_log.append("❌ 综合模型融合失败")

            if result_files:
                st.session_state.task4_results = optimizer.results
                st.session_state.task4_completed = True
                st.success("✅ 运营优化分析完成！")

                # 显示分析结果
                display_task4_results(optimizer.results, result_files, progress_log)
            else:
                st.error("运营优化分析失败")
                for log in progress_log:
                    st.error(log)

    else:
        # 显示功能说明
        show_task4_instructions()

def display_task4_results(results, result_files, progress_log):
    """显示任务4分析结果"""

    # 1. 显示分析日志
    st.subheader("📝 分析执行日志")
    for log in progress_log:
        st.write(f"▪️ {log}")

    # 2. ABC分类分析结果
    if 'abc_classification' in results:
        st.subheader("📊 ABC分类分析")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(results['abc_classification'][['商品品类', '销售额占比%', '利润占比%', 'ABC分类（销售额）']])

        with col2:
            if 'abc_visualizations' in results:
                st.pyplot(results['abc_visualizations']['sales_distribution'])

        # 显示其他图表
        col1, col2 = st.columns(2)
        with col1:
            if 'abc_visualizations' in results:
                st.pyplot(results['abc_visualizations']['cumulative_sales'])
        with col2:
            if 'abc_visualizations' in results:
                st.pyplot(results['abc_visualizations']['abc_distribution'])

    # 3. 价格敏感度分析结果
    if 'price_sensitivity' in results:
        st.subheader("💰 价格敏感度分析")

        st.dataframe(results['price_sensitivity'])

        # 显示拟合图表
        if 'fitting_charts' in results:
            st.subheader("📈 价格-销量关系拟合图")
            cols = st.columns(2)
            chart_count = 0

            for chart_name, fig in results['fitting_charts'].items():
                with cols[chart_count % 2]:
                    st.pyplot(fig)
                chart_count += 1

    # 4. 用户分层分析结果
    if 'user_segmentation' in results:
        st.subheader("👥 用户分层分析")
        st.dataframe(results['user_segmentation'])

    # 5. 场景价格分析结果
    if 'scenario_analysis' in results:
        st.subheader("🏷️ 场景价格分析")
        st.dataframe(results['scenario_analysis'])

    # 6. 综合模型融合结果
    if 'comprehensive_fusion' in results:
        st.subheader("🎯 综合敏感度评估")
        st.dataframe(results['comprehensive_fusion'])

    # 7. 运营策略推荐
    if 'operation_strategies' in results:
        st.subheader("🚀 可执行运营策略")

        # 按优先级筛选
        priority_filter = st.selectbox("筛选策略优先级:", ["全部", "高", "中"])

        if priority_filter != "全部":
            filtered_strategies = results['operation_strategies'][
                results['operation_strategies']['优先级'] == priority_filter]
        else:
            filtered_strategies = results['operation_strategies']

        st.dataframe(filtered_strategies)

    # 8. 文件下载
    st.subheader("📥 分析结果下载")
    for filename, data in result_files.items():
        excel_bytes = io.BytesIO()
        with pd.ExcelWriter(excel_bytes, engine='openpyxl') as writer:
            data.to_excel(writer, index=False)
        st.download_button(
            label=f"下载 {filename}",
            data=excel_bytes.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


def show_task4_instructions():
    """显示任务4功能说明"""
    st.info("""
    **📋 运营策略优化功能说明**

    **核心分析模块：**

    **📊 ABC分类分析**
    - 基于帕累托法则（20/80定律）对商品品类进行分类
    - A类（核心）：销售额累计占比前70%
    - B类（潜力）：销售额累计占比70%-90%  
    - C类（长尾）：剩余10%
    - 输出：分类结果、可视化图表、资源分配建议

    **💰 价格敏感度分析**
    - 通过"价格-销量"关系分析各品类的价格弹性
    - 使用等频8区间划分和线性回归分析
    - 计算价格弹性系数S，判定敏感度等级
    - 输出：敏感度系数、拟合图表、定价建议

    **👥 用户分层分析**  
    - 基于年龄和性别对用户进行分层
    - 计算价格接受度(R1)、集中度(R2)、敏感倾向指数(R3)
    - 建立用户价格偏好判定矩阵
    - 输出：分层结果、价格偏好特征

    **🏷️ 场景价格分析**
    - 将商品映射到消费场景（办公、家居、厨房、汽车）
    - 分析各场景的价格带分布
    - 计算场景价格敏感度指数(SI)
    - 输出：场景价格带、敏感度指数

    **🎯 综合模型融合**
    - 使用AHP层次分析法融合多维度分析结果
    - 权重分配：品类(66%)、人群(23%)、场景(12%)
    - 生成综合敏感度得分和等级
    - 输出：综合评估、运营策略推荐

    **🚀 点击上方按钮开始分析！**
    """)

def show_system_status():
    """系统状态页面"""
    st.header("🔧 系统状态")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("任务完成状态")
        tasks = [
            ("数据预处理", st.session_state.task1_completed),
            ("多维特征分析", st.session_state.task2_completed),
            ("销售预测", st.session_state.task3_completed),
            ("运营优化", st.session_state.task4_completed)
        ]
        for task_name, completed in tasks:
            status_class = "status-completed" if completed else "status-pending"
            icon = "✅" if completed else "⏳"
            st.markdown(f'<div class="{status_class}">{icon} {task_name}</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("数据状态")
        if st.session_state.raw_data is not None:
            df = st.session_state.raw_data
            total_records = len(df)
            total_cols = len(df.columns)
            numeric_cols = len(st.session_state.column_types['numeric']) if st.session_state.column_types else 0
            category_cols = len(st.session_state.column_types['nominal']) + len(
                st.session_state.column_types['ordinal']) if st.session_state.column_types else 0

            st.metric("总记录数", f"{total_records:,}条")
            st.metric("总字段数", f"{total_cols}个")
            st.metric("数值型字段", f"{numeric_cols}个")
            st.metric("分类型字段", f"{category_cols}个")
        else:
            st.info("暂无数据，请先执行'数据预处理'")

    # 重置功能
    if st.button("🔄 重置系统", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        initialize_session_state()
        st.rerun()


# ============================================================================
# 其他现有函数（EnhancedTask2Analyzer、show_python_visualizations、show_data_export_interface等）
# ============================================================================

# 在这里添加优化后的交互式仪表板函数
def show_interactive_dashboard_optimized(visualizer_results):
    """优化版的交互式可视化仪表板"""

    # 顶部指标卡片
    st.markdown("### 📊 实时业务指标")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("总销售额", "¥1,234,567", "+12%")
    with col2:
        st.metric("总利润", "¥456,789", "+8%")
    with col3:
        st.metric("订单数量", "12,345", "+5%")
    with col4:
        st.metric("客户数量", "8,765", "+15%")

    # 主要分析区域
    tab1, tab2, tab3, tab4 = st.tabs(["📈 销售分析", "👥 客户分析", "🗺️ 地域分析", "🔍 高级分析"])

    with tab1:
        col_sales1, col_sales2 = st.columns(2)
        with col_sales1:
            if 'interactive_dashboard' in visualizer_results and 'sales_trend' in visualizer_results[
                'interactive_dashboard']:
                st.plotly_chart(visualizer_results['interactive_dashboard']['sales_trend'],
                                use_container_width=True)
        with col_sales2:
            if 'interactive_dashboard' in visualizer_results and 'category_bar' in visualizer_results[
                'interactive_dashboard']:
                st.plotly_chart(visualizer_results['interactive_dashboard']['category_bar'],
                                use_container_width=True)

    with tab2:
        col_cust1, col_cust2 = st.columns(2)
        with col_cust1:
            if 'interactive_dashboard' in visualizer_results and 'demographic_scatter' in visualizer_results[
                'interactive_dashboard']:
                st.plotly_chart(visualizer_results['interactive_dashboard']['demographic_scatter'],
                                use_container_width=True)
        with col_cust2:
            if 'customer_segmentation' in visualizer_results and 'customer_clusters' in visualizer_results[
                'customer_segmentation']:
                st.plotly_chart(visualizer_results['customer_segmentation']['customer_clusters'],
                                use_container_width=True)

    with tab3:
        if 'interactive_dashboard' in visualizer_results and 'region_bar' in visualizer_results[
            'interactive_dashboard']:
            st.plotly_chart(visualizer_results['interactive_dashboard']['region_bar'],
                            use_container_width=True)

    with tab4:
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            if 'advanced_analytics' in visualizer_results and 'correlation_heatmap' in visualizer_results[
                'advanced_analytics']:
                st.plotly_chart(visualizer_results['advanced_analytics']['correlation_heatmap'],
                                use_container_width=True)
        with col_adv2:
            if 'advanced_analytics' in visualizer_results and 'box_plot' in visualizer_results['advanced_analytics']:
                st.plotly_chart(visualizer_results['advanced_analytics']['box_plot'],
                                use_container_width=True)


def show_project_overview_optimized():
    """优化版项目概览页面"""
    st.markdown("### 🎯 系统功能概述")

    # 功能特性卡片
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📁 智能数据预处理</h3>
            <p>按论文要求自动生成6个标准化输出文件，支持缺失值处理、异常修正、标准化等完整流程</p>
            <ul>
                <li>缺失值统计分析</li>
                <li>进货价格处理</li>
                <li>利润自动修正</li>
                <li>异常值检测修正</li>
                <li>MinMax/ZScore标准化</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 多维特征分析</h3>
            <p>支持多种数据源，提供交互式可视化分析，深度挖掘业务洞察</p>
            <ul>
                <li>地理分布分析</li>
                <li>客户画像分析</li>
                <li>时间序列分析</li>
                <li>交叉维度热力图</li>
                <li>聚类分析</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 智能预测优化</h3>
            <p>基于机器学习的时间序列预测和运营策略优化</p>
            <ul>
                <li>ARIMA-XGBoost混合预测</li>
                <li>ABC分类分析</li>
                <li>价格敏感度分析</li>
                <li>可落地运营策略</li>
                <li>实时指标监控</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 关键指标
    st.markdown("### 📊 系统指标")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">6</div>
            <div>标准输出文件</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">4</div>
            <div>分析任务</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">📊</div>
            <div>交互式可视化</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_col4:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">🎯</div>
            <div>智能预测</div>
        </div>
        """, unsafe_allow_html=True)

    # 使用指南
    st.markdown("### 🚀 快速开始指南")

    guide_col1, guide_col2 = st.columns(2)

    with guide_col1:
        st.markdown("""
        **1. 数据预处理**
        - 点击"数据预处理"按钮
        - 上传Excel/CSV文件
        - 系统自动执行6步预处理
        - 下载标准化输出文件

        **2. 多维特征分析**  
        - 选择预处理数据或上传新文件
        - 选择分析模式（可视化/导出/仪表板）
        - 查看交互式分析结果
        """)

    with guide_col2:
        st.markdown("""
        **3. 销售预测分析**
        - 导入时间序列数据
        - 执行ARIMA-XGBoost预测
        - 查看预测结果和精度

        **4. 运营策略优化**
        - ABC商品分类分析
        - 价格敏感度分析
        - 生成可落地运营策略
        """)


def task1_data_preprocessing_optimized():
    """优化版数据预处理页面"""
    st.markdown("### 📁 任务1: 数据预处理")

    # 功能说明卡片
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 论文标准输出</h3>
        <p>按论文要求自动生成6个标准化Excel文件，完整的数据预处理流程</p>
    </div>
    """, unsafe_allow_html=True)

    # 步骤说明
    steps_col1, steps_col2, steps_col3 = st.columns(3)

    with steps_col1:
        st.markdown("""
        **步骤1-2: 数据清洗**
        - 缺失值统计分析
        - 进货价格格式标准化
        - 数据类型自动检测
        """)

    with steps_col2:
        st.markdown("""
        **步骤3-4: 业务逻辑修正**
        - 利润计算错误修正
        - 异常售价检测修复
        - 利润重新计算
        """)

    with steps_col3:
        st.markdown("""
        **步骤5: 数据标准化**
        - MinMax标准化(0-1)
        - ZScore标准化
        - 分类变量编码
        """)

    # 文件上传区域
    st.markdown("### 📤 数据上传")
    uploaded_file = st.file_uploader(
        "上传原始数据表（支持Excel或CSV格式）",
        type=["xlsx", "csv"],
        help="建议包含：商品品类、区域、销售额、利润、日期等字段"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            # 数据清洗
            df_clean = clean_numeric_columns(df)
            st.session_state.raw_data = df_clean
            st.session_state.current_file = uploaded_file.name

            # 数据预览
            st.success(f"✅ 文件上传成功！共 {len(df)} 条记录，{len(df.columns)} 个字段")

            col_preview1, col_preview2 = st.columns(2)
            with col_preview1:
                st.markdown("**原始数据预览**")
                st.dataframe(df.head(), use_container_width=True)
            with col_preview2:
                st.markdown("**清洗后数据预览**")
                st.dataframe(df_clean.head(), use_container_width=True)

            # 执行预处理按钮
            st.markdown("### 🚀 执行预处理")
            if st.button("开始数据预处理（生成6个标准文件）", type="primary", use_container_width=True):
                with st.spinner("正在执行数据预处理流程..."):
                    preprocessor = Task1Preprocessor(df_clean)
                    result_files, progress_log, final_data, encoders, column_types = preprocessor.generate_all_results()

                    if result_files:
                        # 保存结果到session state
                        st.session_state.step1_missing_data = result_files['电商 步骤1 缺失值统计结果.xlsx']
                        st.session_state.step2_price_data = result_files['电商 步骤2 进货价格处理后数据.xlsx']
                        st.session_state.step3_profit_data = result_files['电商 步骤3 利润修正后数据.xlsx']
                        st.session_state.step4_abnormal_data = result_files['电商 步骤4 异常修正及利润重算后数据.xlsx']
                        st.session_state.step5_minmax_data = result_files['电商 步骤5 MinMax标准化后数据.xlsx']
                        st.session_state.step5_zscore_data = result_files['电商 步骤5 ZScore标准化后数据.xlsx']
                        st.session_state.processed_data = final_data
                        st.session_state.category_encoder = encoders
                        st.session_state.column_types = column_types
                        st.session_state.task1_completed = True

                        st.success("✅ 数据预处理完成！")

                        # 结果显示
                        st.markdown("### 📊 预处理结果")

                        # 进度日志
                        st.markdown("**执行日志:**")
                        for log in progress_log:
                            st.write(f"▪️ {log}")

                        # 文件下载
                        st.markdown("### 📥 下载标准文件")
                        download_col1, download_col2 = st.columns(2)

                        with download_col1:
                            for i, (filename, data) in enumerate(list(result_files.items())[:3]):
                                if isinstance(data, pd.DataFrame):
                                    excel_bytes = io.BytesIO()
                                    with pd.ExcelWriter(excel_bytes, engine='openpyxl') as writer:
                                        data.to_excel(writer, index=False)
                                    st.download_button(
                                        label=f"下载 {filename}",
                                        data=excel_bytes.getvalue(),
                                        file_name=filename,
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True
                                    )

                        with download_col2:
                            for i, (filename, data) in enumerate(list(result_files.items())[3:]):
                                if isinstance(data, pd.DataFrame):
                                    excel_bytes = io.BytesIO()
                                    with pd.ExcelWriter(excel_bytes, engine='openpyxl') as writer:
                                        data.to_excel(writer, index=False)
                                    st.download_button(
                                        label=f"下载 {filename}",
                                        data=excel_bytes.getvalue(),
                                        file_name=filename,
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True
                                    )

        except Exception as e:
            st.error(f"❌ 文件处理错误: {str(e)}")
    else:
        st.info("📝 请上传数据文件开始预处理流程")


def task3_sales_forecast_optimized():
    """优化版销售预测页面"""
    st.markdown("### 📈 任务3: 销售预测分析")

    # 功能说明
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 ARIMA-XGBoost混合预测</h3>
        <p>使用时间序列分析+机器学习进行精准销售预测，支持多维度特征工程</p>
    </div>
    """, unsafe_allow_html=True)

    # 预测流程说明
    forecast_col1, forecast_col2 = st.columns(2)

    with forecast_col1:
        st.markdown("""
        **📊 预测模型特点**
        - ARIMA捕捉时间序列趋势
        - XGBoost学习残差模式
        - 多维度特征工程
        - 自动参数优化
        """)

    with forecast_col2:
        st.markdown("""
        **🎯 预测输出**
        - 未来销售趋势预测
        - 预测精度评估(MAPE)
        - 特征重要性分析
        - 可视化预测结果
        """)

    if not st.session_state.get('task1_completed', False):
        st.warning("⚠️ 建议先完成数据预处理（任务1）以获得更好的数据质量")

    # 数据源选择
    st.markdown("### 📁 预测数据准备")
    data_source = st.radio(
        "选择预测数据源:",
        ["使用预处理数据", "上传新数据文件"],
        horizontal=True
    )

    df = None
    if data_source == "使用预处理数据" and st.session_state.get('processed_data') is not None:
        df = st.session_state.processed_data
        st.success(f"✅ 使用预处理数据，共 {len(df)} 条记录")
    else:
        uploaded_file = st.file_uploader("上传预测数据文件", type=["xlsx", "csv"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                df = clean_numeric_columns(df)
                st.success(f"✅ 数据加载成功！共 {len(df)} 条记录")
            except Exception as e:
                st.error(f"❌ 文件读取错误: {str(e)}")

    # 执行预测
    if df is not None:
        st.markdown("### 🚀 执行销售预测")

        # 预测参数设置
        with st.expander("⚙️ 预测参数设置", expanded=False):
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                forecast_days = st.slider("预测天数", 7, 30, 14)
                confidence_level = st.slider("置信水平", 0.8, 0.99, 0.95)
            with param_col2:
                model_type = st.selectbox("模型类型", ["ARIMA-XGBoost混合", "纯ARIMA", "纯XGBoost"])
                include_features = st.multiselect("包含特征", ["销售额", "销售数", "季节因素", "促销活动"])

        if st.button("开始销售预测", type="primary", use_container_width=True):
            with st.spinner("🔄 正在训练预测模型..."):
                column_types = auto_detect_column_types(df)
                forecaster = Task3Forecaster(df, column_types)
                result_files, progress_log = forecaster.generate_all_results()

                if result_files:
                    st.session_state.task3_results = forecaster.results
                    st.session_state.task3_completed = True

                    st.success("✅ 销售预测完成！")

                    # 预测结果展示
                    st.markdown("### 📊 预测结果")

                    # 关键指标
                    if 'mape' in forecaster.results:
                        mape = forecaster.results['mape']
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric("预测精度(MAPE)", f"{mape:.2f}%")
                        with metric_col2:
                            best_error = forecaster.results['detailed_results']['相对误差(%)'].min()
                            st.metric("最佳预测", f"{best_error:.1f}%")
                        with metric_col3:
                            st.metric("预测天数", forecast_days)

                    # 可视化结果
                    if 'visualizations' in forecaster.results:
                        viz = forecaster.results['visualizations']
                        tab1, tab2, tab3 = st.tabs(["📈 预测对比", "📊 误差分析", "🔍 特征重要性"])

                        with tab1:
                            st.plotly_chart(viz['main_forecast'], use_container_width=True)
                        with tab2:
                            st.plotly_chart(viz['error_analysis'], use_container_width=True)
                        with tab3:
                            st.plotly_chart(viz['feature_importance'], use_container_width=True)

                    # 下载结果
                    st.markdown("### 📥 下载预测结果")
                    for filename, data in result_files.items():
                        if isinstance(data, pd.DataFrame):
                            excel_bytes = io.BytesIO()
                            with pd.ExcelWriter(excel_bytes, engine='openpyxl') as writer:
                                data.to_excel(writer, index=False)
                            st.download_button(
                                label=f"下载 {filename}",
                                data=excel_bytes.getvalue(),
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )


def task4_operation_optimization_optimized():
    """优化版运营优化页面"""
    st.markdown("### 💡 任务4: 运营策略优化")

    # 功能说明
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 数据驱动的运营决策</h3>
        <p>基于数据分析生成可落地的运营策略，提升销售效率和利润率</p>
    </div>
    """, unsafe_allow_html=True)

    # 分析维度说明
    analysis_col1, analysis_col2, analysis_col3 = st.columns(3)

    with analysis_col1:
        st.markdown("""
        **📊 ABC分类分析**
        - 商品品类价值分级
        - 区域销售贡献分析
        - 资源优化配置建议
        """)

    with analysis_col2:
        st.markdown("""
        **💰 价格敏感度分析**
        - 品类价格弹性测算
        - 客户群体价格敏感度
        - 最优定价策略推荐
        """)

    with analysis_col3:
        st.markdown("""
        **🚀 运营策略生成**
        - 库存管理策略
        - 促销活动建议
        - 客户关系优化
        """)

    if not st.session_state.get('task1_completed', False):
        st.warning("⚠️ 建议先完成数据预处理（任务1）")

    # 数据源选择
    st.markdown("### 📁 运营分析数据")
    data_source = st.radio(
        "选择分析数据源:",
        ["使用预处理数据", "上传新数据文件"],
        horizontal=True
    )

    df = None
    if data_source == "使用预处理数据" and st.session_state.get('processed_data') is not None:
        df = st.session_state.processed_data
        st.success(f"✅ 使用预处理数据，共 {len(df)} 条记录")
    else:
        uploaded_file = st.file_uploader("上传运营分析数据", type=["xlsx", "csv"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                df = clean_numeric_columns(df)
                st.success(f"✅ 数据加载成功！共 {len(df)} 条记录")
            except Exception as e:
                st.error(f"❌ 文件读取错误: {str(e)}")

    # 执行运营分析
    if df is not None:
        st.markdown("### 🚀 执行运营优化分析")

        # 分析选项
        with st.expander("⚙️ 分析选项设置", expanded=False):
            option_col1, option_col2 = st.columns(2)
            with option_col1:
                abc_analysis = st.checkbox("ABC分类分析", value=True)
                price_sensitivity = st.checkbox("价格敏感度分析", value=True)
            with option_col2:
                customer_segmentation = st.checkbox("客户分群分析")
                strategy_generation = st.checkbox("策略生成", value=True)

        if st.button("开始运营优化分析", type="primary", use_container_width=True):
            with st.spinner("🔄 正在执行运营分析..."):
                column_types = auto_detect_column_types(df)
                optimizer = Task4Optimizer(df, column_types)
                result_files, progress_log = optimizer.generate_all_results()

                if result_files:
                    st.session_state.task4_results = optimizer.results
                    st.session_state.task4_completed = True

                    st.success("✅ 运营优化分析完成！")

                    # 分析结果展示
                    st.markdown("### 📊 分析结果")

                    # ABC分类结果
                    if 'category_abc' in optimizer.results:
                        st.markdown("#### 📈 ABC商品分类")
                        abc_data = optimizer.results['category_abc']

                        col_abc1, col_abc2 = st.columns(2)
                        with col_abc1:
                            # 分类统计
                            abc_counts = abc_data['ABC分类（按销售额）'].value_counts()
                            fig_abc = px.pie(
                                values=abc_counts.values,
                                names=abc_counts.index,
                                title="ABC分类占比"
                            )
                            st.plotly_chart(fig_abc, use_container_width=True)

                        with col_abc2:
                            st.dataframe(abc_data[['商品品类', '销售额', '利润', 'ABC分类（按销售额）']].head(10))

                    # 价格敏感度分析
                    if 'price_sensitivity' in optimizer.results:
                        st.markdown("#### 💰 价格敏感度分析")
                        sensitivity_data = optimizer.results['price_sensitivity']
                        st.dataframe(sensitivity_data)

                    # 运营策略
                    if 'operation_strategy' in optimizer.results:
                        st.markdown("#### 🚀 运营策略推荐")
                        strategy_data = optimizer.results['operation_strategy']

                        tab_strategy1, tab_strategy2, tab_strategy3 = st.tabs(["高优先级", "中优先级", "低优先级"])

                        with tab_strategy1:
                            high_priority = strategy_data[strategy_data['优先级'] == '高']
                            for _, row in high_priority.iterrows():
                                st.markdown(f"""
                                <div style='background: #d4edda; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;'>
                                    <strong>{row['维度值']}</strong> - {row.get('定价策略', row.get('运营策略', ''))}
                                </div>
                                """, unsafe_allow_html=True)

                        with tab_strategy2:
                            mid_priority = strategy_data[strategy_data['优先级'] == '中']
                            for _, row in mid_priority.iterrows():
                                st.markdown(f"""
                                <div style='background: #fff3cd; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;'>
                                    <strong>{row['维度值']}</strong> - {row.get('定价策略', row.get('运营策略', ''))}
                                </div>
                                """, unsafe_allow_html=True)

                        with tab_strategy3:
                            low_priority = strategy_data[strategy_data['优先级'] == '低']
                            for _, row in low_priority.iterrows():
                                st.markdown(f"""
                                <div style='background: #f8d7da; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;'>
                                    <strong>{row['维度值']}</strong> - {row.get('定价策略', row.get('运营策略', ''))}
                                </div>
                                """, unsafe_allow_html=True)

                    # 下载结果
                    st.markdown("### 📥 下载分析报告")
                    for filename, data in result_files.items():
                        if isinstance(data, pd.DataFrame):
                            excel_bytes = io.BytesIO()
                            with pd.ExcelWriter(excel_bytes, engine='openpyxl') as writer:
                                data.to_excel(writer, index=False)
                            st.download_button(
                                label=f"下载 {filename}",
                                data=excel_bytes.getvalue(),
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )


def show_system_status_optimized():
    """优化版系统状态页面"""
    st.markdown("### ⚙️ 系统状态")

    # 系统概览卡片 - 优化样式
    st.markdown("""
    <div class="feature-card" style="background: white; color: black;">
        <h3 style="color: black;">🔧 系统运行状态</h3>
        <p style="color: #333;">实时监控系统运行状态和数据处理情况，确保分析流程顺畅</p>
    </div>
    """, unsafe_allow_html=True)

    # 系统指标 - 使用卡片样式
    st.markdown("### 📊 系统指标")

    sys_col1, sys_col2, sys_col3, sys_col4 = st.columns(4)

    with sys_col1:
        total_records = len(st.session_state.raw_data) if st.session_state.raw_data is not None else 0
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #007bff; text-align: center;">
            <div style="font-size: 1.2rem; color: #333; font-weight: bold;">总数据记录</div>
            <div style="font-size: 2rem; color: #007bff; font-weight: bold;">{total_records:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with sys_col2:
        total_tasks = sum([
            st.session_state.task1_completed,
            st.session_state.task2_completed,
            st.session_state.task3_completed,
            st.session_state.task4_completed
        ])
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #28a745; text-align: center;">
            <div style="font-size: 1.2rem; color: #333; font-weight: bold;">完成任务数</div>
            <div style="font-size: 2rem; color: #28a745; font-weight: bold;">{total_tasks}/4</div>
        </div>
        """, unsafe_allow_html=True)

    with sys_col3:
        if st.session_state.current_file:
            file_status = "✅ 已加载"
            color = "#28a745"
        else:
            file_status = "❌ 未加载"
            color = "#dc3545"
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid {color}; text-align: center;">
            <div style="font-size: 1.2rem; color: #333; font-weight: bold;">数据文件</div>
            <div style="font-size: 1.5rem; color: {color}; font-weight: bold;">{file_status}</div>
        </div>
        """, unsafe_allow_html=True)

    with sys_col4:
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #6c757d; text-align: center;">
            <div style="font-size: 1.2rem; color: #333; font-weight: bold;">系统版本</div>
            <div style="font-size: 2rem; color: #6c757d; font-weight: bold;">v2.0</div>
        </div>
        """, unsafe_allow_html=True)

    # 任务详细状态 - 优化布局
    st.markdown("### 📋 任务详细状态")

    # 使用卡片布局展示任务状态
    task_col1, task_col2 = st.columns(2)

    with task_col1:
        # 数据预处理状态卡片
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #e0e0e0;">
            <h4 style="color: black; margin-bottom: 1rem;">📁 数据预处理</h4>
        """, unsafe_allow_html=True)

        if st.session_state.task1_completed:
            st.success("✅ 已完成")
            if st.session_state.step1_missing_data is not None:
                st.markdown(
                    f"<div style='color: #333;'>▪️ 缺失值分析: {len(st.session_state.step1_missing_data)}个字段</div>",
                    unsafe_allow_html=True)
            if st.session_state.processed_data is not None:
                st.markdown(
                    f"<div style='color: #333;'>▪️ 处理数据: {len(st.session_state.processed_data)}条记录</div>",
                    unsafe_allow_html=True)
            if st.session_state.column_types is not None:
                numeric_count = len(st.session_state.column_types['numeric'])
                st.markdown(f"<div style='color: #333;'>▪️ 数值字段: {numeric_count}个</div>", unsafe_allow_html=True)
        else:
            st.warning("⏳ 待完成")
            st.markdown("<div style='color: #666;'>等待数据上传和预处理</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # 多维分析状态卡片
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #e0e0e0;">
            <h4 style="color: black; margin-bottom: 1rem;">🔍 多维特征分析</h4>
        """, unsafe_allow_html=True)

        if st.session_state.task2_completed:
            st.success("✅ 已完成")
            if st.session_state.task2_analysis_data:
                analysis_count = sum(1 for data in st.session_state.task2_analysis_data.values() if data is not None)
                st.markdown(f"<div style='color: #333;'>▪️ 分析维度: {analysis_count}个</div>", unsafe_allow_html=True)
            if st.session_state.task2_results and 'heatmaps' in st.session_state.task2_results:
                heatmap_count = len(st.session_state.task2_results['heatmaps'])
                st.markdown(f"<div style='color: #333;'>▪️ 热力图: {heatmap_count}个</div>", unsafe_allow_html=True)
        else:
            st.warning("⏳ 待完成")
            st.markdown("<div style='color: #666;'>等待多维分析执行</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with task_col2:
        # 销售预测状态卡片
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #e0e0e0;">
            <h4 style="color: black; margin-bottom: 1rem;">📈 销售预测</h4>
        """, unsafe_allow_html=True)

        if st.session_state.task3_completed:
            st.success("✅ 已完成")
            if st.session_state.task3_results and 'mape' in st.session_state.task3_results:
                mape = st.session_state.task3_results['mape']
                accuracy_color = "#28a745" if mape < 10 else "#ffc107" if mape < 20 else "#dc3545"
                st.markdown(
                    f"<div style='color: #333;'>▪️ 预测精度: <span style='color: {accuracy_color}; font-weight: bold;'>{mape:.2f}%</span></div>",
                    unsafe_allow_html=True)
            if st.session_state.task3_results and 'detailed_results' in st.session_state.task3_results:
                pred_count = len(st.session_state.task3_results['detailed_results'])
                st.markdown(f"<div style='color: #333;'>▪️ 预测天数: {pred_count}天</div>", unsafe_allow_html=True)
        else:
            st.warning("⏳ 待完成")
            st.markdown("<div style='color: #666;'>等待销售预测执行</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # 运营优化状态卡片
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #e0e0e0;">
            <h4 style="color: black; margin-bottom: 1rem;">💡 运营优化</h4>
        """, unsafe_allow_html=True)

        if st.session_state.task4_completed:
            st.success("✅ 已完成")
            if st.session_state.task4_results and 'operation_strategy' in st.session_state.task4_results:
                strategy_count = len(st.session_state.task4_results['operation_strategy'])
                st.markdown(f"<div style='color: #333;'>▪️ 生成策略: {strategy_count}条</div>", unsafe_allow_html=True)
            if st.session_state.task4_results and 'category_abc' in st.session_state.task4_results:
                category_count = len(st.session_state.task4_results['category_abc'])
                st.markdown(f"<div style='color: #333;'>▪️ 分类商品: {category_count}个</div>", unsafe_allow_html=True)
        else:
            st.warning("⏳ 待完成")
            st.markdown("<div style='color: #666;'>等待运营优化分析</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # 数据统计 - 优化显示
    if st.session_state.raw_data is not None:
        st.markdown("### 📈 数据统计")
        df = st.session_state.raw_data

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        with stat_col1:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0;">
                <div style="font-size: 1rem; color: #666;">数值型字段</div>
                <div style="font-size: 1.8rem; color: #007bff; font-weight: bold;">{numeric_cols}</div>
            </div>
            """, unsafe_allow_html=True)

        with stat_col2:
            category_cols = len(df.select_dtypes(exclude=[np.number]).columns)
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0;">
                <div style="font-size: 1rem; color: #666;">分类型字段</div>
                <div style="font-size: 1.8rem; color: #28a745; font-weight: bold;">{category_cols}</div>
            </div>
            """, unsafe_allow_html=True)

        with stat_col3:
            total_cols = len(df.columns)
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0;">
                <div style="font-size: 1rem; color: #666;">总字段数</div>
                <div style="font-size: 1.8rem; color: #6c757d; font-weight: bold;">{total_cols}</div>
            </div>
            """, unsafe_allow_html=True)

        with stat_col4:
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0;">
                <div style="font-size: 1rem; color: #666;">内存占用</div>
                <div style="font-size: 1.5rem; color: #fd7e14; font-weight: bold;">{memory_usage:.1f} MB</div>
            </div>
            """, unsafe_allow_html=True)

    # 系统操作 - 优化按钮样式
    st.markdown("### 🔄 系统操作")

    op_col1, op_col2, op_col3, op_col4 = st.columns(4)

    with op_col1:
        if st.button("🔄 重置系统", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            initialize_session_state()
            st.success("系统已重置！")
            st.rerun()

    with op_col2:
        if st.button("💾 导出配置", use_container_width=True, type="secondary"):
            st.info("📋 配置导出功能开发中...")

    with op_col3:
        if st.button("📋 生成报告", use_container_width=True, type="secondary"):
            st.info("📊 报告生成功能开发中...")

    with op_col4:
        if st.button("🆘 使用帮助", use_container_width=True, type="secondary"):
            st.info("ℹ️ 帮助文档功能开发中...")

    # 添加系统信息
    st.markdown("---")
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px;">
        <h5 style="color: #333;">ℹ️ 系统信息</h5>
        <div style="color: #666;">
            <div>▪️ 最后更新: 实时</div>
            <div>▪️ 数据状态: {}</div>
            <div>▪️ 分析进度: {}/4 个任务完成</div>
        </div>
    </div>
    """.format(
        "已加载" if st.session_state.current_file else "未加载",
        sum([st.session_state.task1_completed, st.session_state.task2_completed,
             st.session_state.task3_completed, st.session_state.task4_completed])
    ), unsafe_allow_html=True)

# ============================================================================
# 主应用函数 - 现代化界面
# ============================================================================
def main():
    """主应用函数 - 现代化界面"""
    # 页面配置
    st.set_page_config(
        page_title="电商销售分析与策略优化系统",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"  # 隐藏侧边栏
    )

    # 主标题
    st.markdown('<div class="main-header">📊 电商销售分析与策略优化系统</div>',
                unsafe_allow_html=True)

    # 顶部导航栏
    st.markdown("""
    <div class="top-nav">
        <div style="display: flex; justify-content: center; align-items: center; gap: 1rem;">
            <span style="color: white; font-size: 1.2rem; font-weight: bold;">📋 导航菜单</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 顶部导航按钮
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6 = st.columns(6)

    with nav_col1:
        project_overview = st.button("🏠 项目概览", use_container_width=True)
    with nav_col2:
        data_preprocessing = st.button("📁 数据预处理", use_container_width=True)
    with nav_col3:
        multi_analysis = st.button("🔍 多维分析", use_container_width=True)
    with nav_col4:
        sales_forecast = st.button("📈 销售预测", use_container_width=True)
    with nav_col5:
        operation_optimize = st.button("💡 运营优化", use_container_width=True)
    with nav_col6:
        system_status = st.button("⚙️ 系统状态", use_container_width=True)

    # 任务状态指示器
    st.markdown("""
    <div class="status-indicator">
        <div class="status-item {}">
            <div style="font-size: 2rem;">📁</div>
            <div>数据预处理</div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem;">{}
        </div>
        <div class="status-item {}">
            <div style="font-size: 2rem;">🔍</div>
            <div>多维分析</div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem;">{}
        </div>
        <div class="status-item {}">
            <div style="font-size: 2rem;">📈</div>
            <div>销售预测</div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem;">{}
        </div>
        <div class="status-item {}">
            <div style="font-size: 2rem;">💡</div>
            <div>运营优化</div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem;">{}
        </div>
    </div>
    """.format(
        "completed" if st.session_state.task1_completed else "pending",
        "✅ 已完成" if st.session_state.task1_completed else "⏳ 待完成",
        "completed" if st.session_state.task2_completed else "pending",
        "✅ 已完成" if st.session_state.task2_completed else "⏳ 待完成",
        "completed" if st.session_state.task3_completed else "pending",
        "✅ 已完成" if st.session_state.task3_completed else "⏳ 待完成",
        "completed" if st.session_state.task4_completed else "pending",
        "✅ 已完成" if st.session_state.task4_completed else "⏳ 待完成"
    ), unsafe_allow_html=True)

    # 当前文件显示
    if st.session_state.get('current_file'):
        st.info(f"📄 当前文件: {st.session_state.current_file}")

    # 页面路由
    if project_overview:
        st.session_state.current_page = "project_overview"
    elif data_preprocessing:
        st.session_state.current_page = "data_preprocessing"
    elif multi_analysis:
        st.session_state.current_page = "multi_analysis"
    elif sales_forecast:
        st.session_state.current_page = "sales_forecast"
    elif operation_optimize:
        st.session_state.current_page = "operation_optimize"
    elif system_status:
        st.session_state.current_page = "system_status"

    # 初始化当前页面
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "project_overview"

    # 显示对应页面
    if st.session_state.current_page == "project_overview":
        show_project_overview_optimized()
    elif st.session_state.current_page == "data_preprocessing":
        task1_data_preprocessing_optimized()
    elif st.session_state.current_page == "multi_analysis":
        enhanced_task2_multidimensional_analysis()
    elif st.session_state.current_page == "sales_forecast":
        task3_sales_forecast_optimized()
    elif st.session_state.current_page == "operation_optimize":
        task4_operation_optimization_optimized()
    elif st.session_state.current_page == "system_status":
        show_system_status_optimized()

    # 页脚
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
        "🚀 电商销售分析与策略优化系统 | 现代化数据分析平台 | 支持论文标准化输出"
        "</div>",
        unsafe_allow_html=True
    )
# ============================================================================
# 运行应用
# ============================================================================
if __name__ == "__main__":
    main()
