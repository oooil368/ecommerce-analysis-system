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
from scipy.stats import linregress

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 页面配置
# ============================================================================

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
        'step1_missing_data': None,  # 步骤1：缺失值统计结果
        'step2_price_data': None,  # 步骤2：进货价格处理后数据
        'step3_profit_data': None,  # 步骤3：利润修正后数据
        'step4_abnormal_data': None,  # 步骤4：异常修正及利润重算后数据
        'step5_minmax_data': None,  # 步骤5：MinMax标准化后数据
        'step5_zscore_data': None,  # 步骤5：ZScore标准化后数据
        'processed_data': None,  # 最终处理数据
        'category_encoder': None,  # 分类变量编码器
        'current_file': None,
        'task1_completed': False,
        'task2_completed': False,
        'task3_completed': False,
        'task4_completed': False,
        'column_types': None  # 字段类型
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
# 任务1：数据预处理类（按论文要求生成标准化输出文件）
# ============================================================================
# ============================================================================
# 任务1：数据预处理类（按论文要求生成标准化输出文件）- 基于源代码重构
# ============================================================================
class Task1Preprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.results = {}
        self.column_types = None

    def step1_missing_value_analysis(self):
        """步骤1: 缺失值统计分析（生成电商 步骤1 缺失值统计结果.xlsx）"""
        # 计算缺失值统计
        rows = len(self.df)
        missing_stats = []

        for col in self.df.columns:
            non_null_count = self.df[col].count()
            missing_count = rows - non_null_count
            missing_rate = (missing_count / rows) * 100

            missing_stats.append({
                '字段名': col,
                '数据类型': str(self.df[col].dtype),
                '非空值数量': non_null_count,
                '缺失值数量': missing_count,
                '缺失比例%': round(missing_rate, 2)
            })

        missing_df = pd.DataFrame(missing_stats)
        self.results['step1_missing_stats'] = missing_df
        return missing_df

    def step2_price_processing(self, missing_stats):
        """步骤2: 进货价格处理（生成电商 步骤2 进货价格处理后数据.xlsx）"""
        import re

        df_step2 = self.df.copy()

        # 处理进货价格字段（基于源代码逻辑）
        if '进货价格' in df_step2.columns:
            # 使用正则表达式去除非数字和非小数点字符，转换为数值型
            df_step2['进货价格'] = df_step2['进货价格'].apply(
                lambda x: float(re.sub(r'[^\d\.]', '', str(x))) if re.search(r'[\d\.]', str(x)) else None
            )
            # 转换为整数型（若存在小数，四舍五入）
            df_step2['进货价格'] = df_step2['进货价格'].round().astype('Int64')  # 使用Int64支持缺失值

        self.results['step2_processed'] = df_step2
        return df_step2

    def step3_profit_correction(self, df_step2):
        """步骤3: 利润修正（生成电商 步骤3 利润修正后数据.xlsx）"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error

        df_step3 = df_step2.copy()

        # 检查必要字段是否存在
        required_cols = ['实际售价', '进货价格', '销售数', '利润']
        missing_cols = [col for col in required_cols if col not in df_step3.columns]
        if missing_cols:
            st.warning(f"利润修正需以下字段：{missing_cols}，数据中缺失，跳过利润修正")
            return df_step3

        # 计算理论利润
        df_step3['理论利润'] = (df_step3['实际售价'] - df_step3['进货价格']) * df_step3['销售数']
        # 筛选错误和正确数据
        error_data = df_step3[df_step3['利润'] != df_step3['理论利润']].copy()
        correct_data = df_step3[df_step3['利润'] == df_step3['理论利润']].copy()

        st.info(f"利润计算错误数据条数：{len(error_data)}")
        st.info(f"利润计算正确数据条数（训练数据）：{len(correct_data)}")

        if len(correct_data) == 0:
            st.warning("无利润计算正确的数据，无法训练模型进行补插，跳过利润修正")
            df_step3 = df_step3.drop(columns='理论利润')
            return df_step3

        # 准备模型训练数据
        features = ['实际售价', '进货价格', '销售数']
        X = correct_data[features]
        y = correct_data['利润']

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 1. 训练随机森林模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        # 评估随机森林模型
        rf_pred_test = rf_model.predict(X_test)
        rf_mse = mean_squared_error(y_test, rf_pred_test)

        # 2. 训练KNN模型
        knn_model = KNeighborsRegressor(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        # 评估KNN模型
        knn_pred_test = knn_model.predict(X_test)
        knn_mse = mean_squared_error(y_test, knn_pred_test)

        # 选择MSE较小的模型进行利润补插
        if rf_mse <= knn_mse:
            st.info(f"选择随机森林模型进行利润补插 (MSE: {rf_mse:.2f})")
            if len(error_data) > 0:
                error_X = error_data[features]
                pred_error = rf_model.predict(error_X)
                # 数据类型转换
                pred_error = pred_error.round().astype(df_step3['利润'].dtype)
                # 重置索引确保对齐
                df_step3 = df_step3.reset_index(drop=True)
                error_data = error_data.reset_index(drop=True)
                # 更新错误利润值（保持列名为"利润"）
                df_step3.loc[error_data.index, '利润'] = pred_error
        else:
            st.info(f"选择KNN模型进行利润补插 (MSE: {knn_mse:.2f})")
            if len(error_data) > 0:
                error_X = error_data[features]
                pred_error = knn_model.predict(error_X)
                # 数据类型转换
                pred_error = pred_error.round().astype(df_step3['利润'].dtype)
                # 重置索引确保对齐
                df_step3 = df_step3.reset_index(drop=True)
                error_data = error_data.reset_index(drop=True)
                # 更新错误利润值（保持列名为"利润"）
                df_step3.loc[error_data.index, '利润'] = pred_error

        # 删除临时的理论利润列
        if '理论利润' in df_step3.columns:
            df_step3 = df_step3.drop(columns='理论利润')

        self.results['step3_processed'] = df_step3
        return df_step3

    def step4_abnormal_correction(self, df_step3):
        """步骤4: 异常值修正及利润重算（生成电商 步骤4 异常修正及利润重算后数据.xlsx）"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error

        df_step4 = df_step3.copy()

        # 检查必要字段是否存在
        required_cols = ['实际售价', '进货价格', '销售数', '客户年龄']
        missing_cols = [col for col in required_cols if col not in df_step4.columns]
        if missing_cols:
            st.warning(f"异常修正需以下字段：{missing_cols}，数据中缺失，跳过异常修正")
            return df_step4

        # 标记异常数据（实际售价 < 进货价格）
        abnormal_mask = df_step4['实际售价'] < df_step4['进货价格']
        abnormal_data = df_step4[abnormal_mask].copy()
        normal_data = df_step4[~abnormal_mask].copy()

        st.info(f"成本高于售价的异常数据条数：{len(abnormal_data)}")
        st.info(f"正常数据条数（训练数据）：{len(normal_data)}")

        if len(normal_data) == 0:
            st.warning("无正常售价数据，无法训练模型进行异常修正，跳过异常修正")
            return df_step4

        # 准备模型训练数据（预测合理实际售价）
        features = ['进货价格', '销售数', '客户年龄']
        target = '实际售价'
        X = normal_data[features]
        y = normal_data[target]

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 1. 训练随机森林模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        # 评估随机森林模型
        rf_pred_test = rf_model.predict(X_test)
        rf_mse = mean_squared_error(y_test, rf_pred_test)

        # 2. 训练KNN模型
        knn_model = KNeighborsRegressor(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        # 评估KNN模型
        knn_pred_test = knn_model.predict(X_test)
        knn_mse = mean_squared_error(y_test, knn_pred_test)

        # 综合两种模型结果进行售价补插（取平均值）
        if len(abnormal_data) > 0:
            abnormal_X = abnormal_data[features]
            rf_pred_abnormal = rf_model.predict(abnormal_X)
            knn_pred_abnormal = knn_model.predict(abnormal_X)
            combined_pred = (rf_pred_abnormal + knn_pred_abnormal) / 2
            # 数据类型转换（确保与原售价字段一致）
            combined_pred = combined_pred.round().astype(df_step4[target].dtype)
            # 更新异常数据的售价
            df_step4.loc[abnormal_mask, target] = combined_pred

        # 二次检查剩余异常（若仍有售价<进货价，将售价设为进货价）
        remaining_abnormal_mask = df_step4['实际售价'] < df_step4['进货价格']
        if remaining_abnormal_mask.sum() > 0:
            st.info(f"二次检查发现{remaining_abnormal_mask.sum()}条剩余异常数据，将售价设为进货价")
            df_step4.loc[remaining_abnormal_mask, '实际售价'] = df_step4.loc[remaining_abnormal_mask, '进货价格']

        # 重新计算正确利润（保持列名为"利润"）
        df_step4['利润'] = (df_step4['实际售价'] - df_step4['进货价格']) * df_step4['销售数']

        self.results['step4_processed'] = df_step4
        return df_step4

    def step5_standardization(self, df_step4):
        """步骤5: 标准化处理（生成电商 步骤5 MinMax标准化后数据.xlsx和ZScore标准化后数据.xlsx）"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        df_original = df_step4.copy()

        # 定义需标准化的数值列（基于源代码逻辑）
        required_cols = ["进货价格", "实际售价", "销售数", "利润"]
        # 若存在销售额列，加入标准化范围
        if "销售额" in df_original.columns:
            required_cols.append("销售额")

        # 检查列是否存在
        missing_cols = [col for col in required_cols if col not in df_original.columns]
        if missing_cols:
            st.warning(f"标准化需以下字段：{missing_cols}，数据中缺失，跳过标准化")
            return df_original, df_original

        # 筛选数值型列（排除非数值数据）
        numeric_cols = [col for col in required_cols if pd.api.types.is_numeric_dtype(df_original[col])]
        if not numeric_cols:
            st.warning("无可用的数值型列进行标准化")
            return df_original, df_original

        st.info(f"待标准化的数值列：{numeric_cols}")

        # 1. Z-Score标准化（均值为0，标准差为1）
        zscore_scaler = StandardScaler()
        df_zscore = df_original.copy()
        df_zscore[numeric_cols] = zscore_scaler.fit_transform(df_zscore[numeric_cols])

        # 2. Min-Max标准化（范围0-1）
        minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        df_minmax = df_original.copy()
        df_minmax[numeric_cols] = minmax_scaler.fit_transform(df_minmax[numeric_cols])

        self.results['step5_minmax'] = df_minmax
        self.results['step5_zscore'] = df_zscore
        self.results['numeric_cols'] = numeric_cols

        return df_minmax, df_zscore

    def generate_all_results(self):
        """生成所有步骤的结果（按论文要求的文件格式）"""
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
                f"步骤2：完成进货价格处理",
                f"步骤3：完成利润修正",
                f"步骤4：完成异常值修正",
                f"步骤5：完成标准化处理，生成MinMax和ZScore两种标准化结果"
            ]

            return result_files, progress_log, final_data, encoders, self.column_types

        except Exception as e:
            return None, [f"预处理错误: {str(e)}"], None, None, None
# ============================================================================
# 增强版任务2：多维销售特征分析类（按论文要求重构）
# ============================================================================
# ============================================================================
# 增强版任务2：多维销售特征分析类（按论文要求重构）- 修复热力图错误
# ============================================================================
class EnhancedTask2Analyzer:
    def __init__(self, df, column_types):
        self.df = df.copy()
        self.column_types = column_types
        self.results = {}

    def create_heatmaps(self):
        """创建热力图 - 修复数据类型问题"""
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

                # 确保利润列是数值类型
                self.df['利润'] = pd.to_numeric(self.df['利润'], errors='coerce')

                # 过滤掉无效数据
                heatmap_data = self.df[['商品品类', '省份', '利润']].dropna()

                if len(heatmap_data) > 0:
                    plt.figure(figsize=(12, 8))
                    category_province_pivot = heatmap_data.pivot_table(
                        index='商品品类',
                        columns='省份',
                        values='利润',
                        aggfunc='sum',
                        fill_value=0
                    )

                    # 确保数据是数值类型
                    category_province_pivot = category_province_pivot.astype(float)

                    # 限制行列数量，避免热力图形状过大
                    if len(category_province_pivot) > 20:
                        category_province_pivot = category_province_pivot.head(20)
                    if len(category_province_pivot.columns) > 15:
                        category_province_pivot = category_province_pivot[category_province_pivot.columns[:15]]

                    sns.heatmap(category_province_pivot, cmap='Blues', annot=False, fmt='.0f')
                    plt.title('商品品类和省份交叉的利润热力图')
                    plt.xlabel('省份')
                    plt.xticks(rotation=45)
                    plt.ylabel('商品品类')
                    plt.tight_layout()
                    figs['category_province_profit'] = plt.gcf()
                    plt.close()
                else:
                    st.warning("商品品类-省份热力图：无有效数据")

            # 2. 省份与日期交叉热力图
            if all(col in self.df.columns for col in ['日期', '省份', '利润']):
                # 确保数据是数值类型
                self.df['利润'] = pd.to_numeric(self.df['利润'], errors='coerce')
                self.df['日期'] = pd.to_numeric(self.df['日期'], errors='coerce')

                # 过滤掉无效数据
                heatmap_data = self.df[['日期', '省份', '利润']].dropna()

                if len(heatmap_data) > 0:
                    plt.figure(figsize=(15, 8))
                    province_date_pivot = heatmap_data.pivot_table(
                        index='省份',
                        columns='日期',
                        values='利润',
                        aggfunc='sum',
                        fill_value=0
                    )

                    # 确保数据是数值类型
                    province_date_pivot = province_date_pivot.astype(float)

                    # 限制行列数量
                    if len(province_date_pivot) > 15:
                        province_date_pivot = province_date_pivot.head(15)
                    if len(province_date_pivot.columns) > 20:
                        province_date_pivot = province_date_pivot[province_date_pivot.columns[:20]]

                    sns.heatmap(province_date_pivot, cmap='Blues', annot=False, fmt='.0f')
                    plt.title('省份和日期交叉的利润热力图')
                    plt.xlabel('日期')
                    plt.xticks(rotation=90)
                    plt.ylabel('省份')
                    plt.tight_layout()
                    figs['province_date_profit'] = plt.gcf()
                    plt.close()
                else:
                    st.warning("省份-日期热力图：无有效数据")

            self.results['heatmaps'] = figs
            return len(figs) > 0

        except Exception as e:
            st.error(f"热力图生成错误: {str(e)}")
            import traceback
            st.error(f"详细错误: {traceback.format_exc()}")
            return False

    def perform_clustering_analysis(self):
        """执行聚类分析 - 修复数据类型问题"""
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
            df_numeric = self.df[existing_numeric_cols].copy()

            # 确保所有列都是数值类型
            for col in existing_numeric_cols:
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

            df_numeric = df_numeric.fillna(0)

            # 检查数据有效性
            if df_numeric.isnull().any().any() or (df_numeric == 0).all().any():
                st.warning("聚类数据包含无效值，跳过聚类分析")
                return False

            # 确定最佳聚类数k
            sse = []
            silhouette_scores = []
            k_range = range(2, min(11, len(df_numeric) // 2))  # 避免k值过大

            for k in k_range:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=2024, n_init='auto')
                    kmeans.fit(df_numeric)
                    sse.append(kmeans.inertia_)
                    labels = kmeans.labels_
                    if len(set(labels)) > 1:  # 确保有多个聚类
                        score = silhouette_score(df_numeric, labels)
                        silhouette_scores.append(score)
                    else:
                        silhouette_scores.append(0)
                except Exception as e:
                    st.warning(f"聚类数k={k}时出错: {e}")
                    sse.append(0)
                    silhouette_scores.append(0)

            # 绘制评估图表
            if len(sse) > 0 and len(silhouette_scores) > 0:
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
                if max(silhouette_scores) > 0:
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
                else:
                    st.warning("无法确定有效的最佳k值，跳过聚类")
                    return False
            else:
                st.warning("聚类评估数据不足，跳过聚类分析")
                return False

        except Exception as e:
            st.error(f"聚类分析错误: {str(e)}")
            import traceback
            st.error(f"详细错误: {traceback.format_exc()}")
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

        # 确保年龄是数值类型
        self.df['客户年龄'] = pd.to_numeric(self.df['客户年龄'], errors='coerce')

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

        # 确保日期是数值类型
        self.df[date_col] = pd.to_numeric(self.df[date_col], errors='coerce')

        time_stats = self.df.groupby(date_col).size().reset_index()
        time_stats.columns = ['日期', '订单人数总和']

        return time_stats

    def generate_correlation_analysis(self):
        """生成相关性分析数据（对应论文图13）"""
        numeric_cols = self.column_types['numeric']
        if len(numeric_cols) < 2:
            return None

        # 确保所有数值列都是数值类型
        correlation_data = self.df[numeric_cols].copy()
        for col in numeric_cols:
            correlation_data[col] = pd.to_numeric(correlation_data[col], errors='coerce')

        correlation_data = correlation_data.dropna()

        if len(correlation_data) < 2:
            return None

        correlation_matrix = correlation_data.corr().round(4)

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
# 任务3：销售预测类（修复版 - 与源代码保持一致）
# ============================================================================
class Task3Forecaster:
    def __init__(self, df, column_types):
        self.df = df.copy()
        self.column_types = column_types
        self.results = {}

    def prepare_time_series_data(self):
        """使用源代码的数据准备逻辑 - 修复数据类型问题"""
        try:
            # 使用源代码的直接转换方式
            date_col = next((col for col in self.column_types['identifier'] if '日期' in col), None)
            if not date_col:
                st.error("未识别到日期字段，无法构建时间序列")
                return False

            # 改为源代码的转换方式
            self.df[date_col] = self.df[date_col].astype(int)

            # 确保利润列是数值类型（源代码方式）
            self.df['利润'] = pd.to_numeric(self.df['利润'], errors='coerce')
            self.df = self.df.dropna(subset=['利润'])

            # 按日聚合利润数据
            daily_profit = self.df.groupby(date_col)['利润'].sum().reset_index()
            daily_profit = daily_profit.rename(columns={'利润': '每日总利润'})

            # 划分训练集和测试集
            train = daily_profit[daily_profit[date_col] <= 24]
            test = daily_profit[daily_profit[date_col] > 24]

            if len(train) == 0 or len(test) == 0:
                st.error("数据日期范围不足，无法划分训练测试集")
                return False

            # 🔥 关键修复：确保数据是标准的numpy数组，不是IntegerArray
            self.results['time_series_data'] = daily_profit
            self.results['train_data'] = train
            self.results['test_data'] = test
            self.results['date_col'] = date_col

            # 转换为标准的numpy数组，避免IntegerArray问题
            self.results['y_train'] = train['每日总利润'].values.astype(float)
            self.results['y_test'] = test['每日总利润'].values.astype(float)

            st.success(f"时间序列准备完成：训练集{len(train)}天，测试集{len(test)}天")
            return True

        except Exception as e:
            st.error(f"时间序列准备错误: {str(e)}")
            return False

    def create_features(self, day_indices, residuals=None):
        """使用源代码的特征工程逻辑"""
        features = []

        # 预计算每个日期的统计量（源代码逻辑）
        daily_stats = self.df.groupby(self.results['date_col']).agg({
            '销售额': ['count', 'mean', 'sum'],
            '实际售价': 'mean',
            '进货价格': 'mean',
            '客户性别': lambda x: (x == '女').mean()
        }).round(4)

        daily_stats.columns = ['order_count', 'avg_sale', 'total_sale',
                               'avg_selling_price', 'avg_cost_price', 'female_ratio']

        # 源代码的毛利率计算（不处理除0）
        daily_stats['gross_profit_margin'] = (
                (daily_stats['avg_selling_price'] - daily_stats['avg_cost_price']) /
                daily_stats['avg_cost_price']
        ).fillna(0).round(4)

        # 源代码的单客价值计算
        daily_stats['customer_value'] = (
                daily_stats['total_sale'] / daily_stats['order_count']
        ).fillna(0).round(2)

        # 训练集统计量（用于填充缺失值）- 源代码逻辑
        train_days_data = self.df[self.df[self.results['date_col']] <= 24]
        train_stats = train_days_data.groupby(self.results['date_col']).agg({
            '销售额': ['count', 'mean', 'sum'],
            '客户性别': lambda x: (x == '女').mean()
        })
        train_stats.columns = ['order_count', 'avg_sale', 'total_sale', 'female_ratio']

        for day in day_indices:
            day_features = {}

            # 1. 基础时间特征（与源代码一致）
            day_features['day'] = int(day)
            day_features['day_of_week'] = (int(day) - 1) % 7
            day_features['day_of_month'] = int(day)
            day_features['is_weekend'] = 1 if day_features['day_of_week'] in [5, 6] else 0
            day_features['is_month_end'] = 1 if int(day) >= 28 else 0

            # 2. 从预计算的统计量中获取业务特征（源代码逻辑）
            if day in daily_stats.index:
                stats = daily_stats.loc[day]
                day_features.update({
                    'order_count': float(stats['order_count']),
                    'avg_sale_amount': float(stats['avg_sale']),
                    'total_sale': float(stats['total_sale']),
                    'gross_profit_margin': float(stats['gross_profit_margin']),
                    'customer_value': float(stats['customer_value']),
                    'female_ratio': float(stats['female_ratio'])
                })
            else:
                # 使用源代码的中位数填充逻辑
                day_features.update({
                    'order_count': float(train_stats['order_count'].median()),
                    'avg_sale_amount': float(train_stats['avg_sale'].median()),
                    'total_sale': float(train_stats['total_sale'].median()),
                    'gross_profit_margin': float(0.3),  # 源代码的默认值
                    'customer_value': float(
                        train_stats['total_sale'].median() / max(1, train_stats['order_count'].median())),
                    'female_ratio': float(train_stats['female_ratio'].median())
                })

            # 3. 滞后残差特征（源代码逻辑）
            if residuals is not None:
                for lag in [1, 2, 3]:
                    lag_day = int(day) - lag
                    lag_key = f'residual_lag_{lag}'
                    if lag_day > 0 and lag_day in residuals.index:
                        day_features[lag_key] = float(residuals[lag_day])
                    else:
                        day_features[lag_key] = float(residuals.median() if not residuals.empty else 0)

            features.append(day_features)

        return pd.DataFrame(features)

    def hybrid_forecast(self):
        """使用源代码的ARIMA-XGBoost混合预测逻辑 - 修复数据类型"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from xgboost import XGBRegressor
            from sklearn.metrics import mean_absolute_percentage_error

            # 获取数据
            train = self.results['train_data']
            test = self.results['test_data']
            y_train = self.results['y_train']  # 已经是float数组
            y_test = self.results['y_test']  # 已经是float数组
            date_col = self.results['date_col']

            # 1. ARIMA建模 - 使用源代码参数 (2,1,2)
            st.info("Step 1: ARIMA建模...")
            try:
                # 🔥 确保y_train是标准的numpy float数组
                y_train_arima = y_train.astype(float)

                arima_model = ARIMA(y_train_arima, order=(2, 1, 2))  # 改为源代码参数
                arima_fit = arima_model.fit()
                arima_train_pred = arima_fit.predict(start=0, end=len(y_train_arima) - 1)
                arima_test_pred = arima_fit.forecast(steps=len(y_test))
                st.success(f"ARIMA模型训练成功 (AIC: {arima_fit.aic:.2f})")
            except Exception as e:
                st.warning(f"ARIMA模型训练失败，使用均值预测: {e}")
                # 使用numpy数组避免数据类型问题
                arima_train_pred = np.full_like(y_train, float(np.mean(y_train)))
                arima_test_pred = np.full_like(y_test, float(np.mean(y_train)))
                arima_fit = None

            # 2. 计算残差 - 确保是float类型
            residuals_train = y_train.astype(float) - arima_train_pred.astype(float)
            residual_series = pd.Series(residuals_train, index=train[date_col].values)

            # 3. XGBoost学习残差 - 使用源代码参数
            st.info("Step 2: XGBoost学习残差...")

            # 创建特征
            X_train = self.create_features(train[date_col].values, residual_series)
            X_train = X_train.fillna(0)

            # 确保特征都是数值类型
            for col in X_train.columns:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_train = X_train.fillna(0)

            # XGBoost模型训练 - 源代码参数
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
            X_test = self.create_features(test[date_col].values)
            X_test = X_test.fillna(0)

            # 确保特征列一致
            for col in X_train.columns:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[X_train.columns]

            # 确保测试集特征都是数值类型
            for col in X_test.columns:
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
            X_test = X_test.fillna(0)

            # 预测残差
            xgb_residual_pred = xgb_model.predict(X_test)

            # 4. 最终预测
            final_pred = arima_test_pred.astype(float) + xgb_residual_pred.astype(float)
            mape = mean_absolute_percentage_error(y_test, final_pred) * 100

            # 保存结果
            self.results['arima_model'] = arima_fit
            self.results['xgb_model'] = xgb_model
            self.results['arima_test_pred'] = arima_test_pred.astype(float)
            self.results['xgb_residual_pred'] = xgb_residual_pred.astype(float)
            self.results['final_pred'] = final_pred.astype(float)
            self.results['mape'] = mape
            self.results['residuals_train'] = residuals_train.astype(float)
            self.results['feature_importance'] = pd.DataFrame({
                'feature': X_train.columns,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)

            # 创建详细结果表
            results_df = pd.DataFrame({
                '日期': test[date_col].values,
                '实际利润': y_test,
                'ARIMA预测': arima_test_pred.astype(float),
                'XGBoost残差预测': xgb_residual_pred.astype(float),
                '最终预测': final_pred.astype(float),
                '相对误差(%)': (np.abs(y_test - final_pred) / y_test * 100).astype(float)
            })
            self.results['detailed_results'] = results_df

            st.success(f"混合预测完成！测试集MAPE: {mape:.2f}%")
            return True

        except Exception as e:
            st.error(f"混合预测错误: {str(e)}")
            import traceback
            st.error(f"详细错误: {traceback.format_exc()}")
            return False

    def generate_visualizations(self):
        """生成可视化图表 - 保持不变"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            figs = {}

            # 1. 主预测对比图
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            train = self.results['train_data']
            test = self.results['test_data']
            date_col = self.results['date_col']
            y_train = self.results['y_train']
            y_test = self.results['y_test']

            # 绘制训练集实际值
            ax1.plot(train[date_col], y_train / 10000, 'bo-', label='训练集实际值',
                     alpha=0.7, markersize=6, linewidth=2)
            # 绘制测试集实际值
            ax1.plot(test[date_col], y_test / 10000, 'ro-', label='测试集实际值',
                     alpha=0.7, markersize=8, linewidth=2)

            # 绘制ARIMA训练集拟合值（源代码中的图表）
            try:
                arima_train_fit = self.results['arima_model'].predict(start=1, end=24)
                ax1.plot(train[date_col], arima_train_fit / 10000, 'c--',
                         label='ARIMA训练集拟合', alpha=0.8, linewidth=2)
            except:
                pass

            # 绘制ARIMA测试集预测值
            ax1.plot(test[date_col], self.results['arima_test_pred'] / 10000, 'm--',
                     label='ARIMA测试集预测', alpha=0.8, linewidth=2)
            # 绘制最终组合预测值
            ax1.plot(test[date_col], self.results['final_pred'] / 10000, 'gs-',
                     label='ARIMA+XGBoost最终预测', markersize=8, linewidth=2)

            ax1.set_xlabel('日期 (11月天数)', fontsize=12)
            ax1.set_ylabel('利润 (万元)', fontsize=12)
            ax1.set_title('利润预测对比图', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=24.5, color='gray', linestyle=':', alpha=0.7, linewidth=2)
            ax1.text(24.7, ax1.get_ylim()[1] * 0.9, '测试集开始', rotation=90, va='top', fontsize=10)
            figs['main_forecast'] = fig1

            # 2. 误差分析图
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            relative_errors = self.results['detailed_results']['相对误差(%)']
            bars = ax2.bar(test[date_col], relative_errors, alpha=0.7, color='orange',
                           edgecolor='darkorange', linewidth=1)

            ax2.set_xlabel('日期 (11月天数)', fontsize=12)
            ax2.set_ylabel('相对误差 (%)', fontsize=12)
            ax2.set_title(f'预测误差分析 (MAPE = {self.results["mape"]:.2f}%)',
                          fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            # 添加误差数值标签
            for date, error in zip(test[date_col], relative_errors):
                ax2.text(date, error + 1, f'{error:.1f}%', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')
            figs['error_analysis'] = fig2

            # 3. 残差分析图
            fig3, ax3 = plt.subplots(figsize=(12, 6))

            if 'residuals_train' in self.results:
                residuals_train = self.results['residuals_train']

                # 绘制残差
                train_dates = train[date_col].values
                ax3.plot(train_dates, residuals_train, 'o-', color='purple',
                         alpha=0.7, markersize=6, linewidth=2, label='每日残差')

                # 零基准线
                ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='零基准线')

                # 均值线
                mean_residual = residuals_train.mean()
                ax3.axhline(y=mean_residual, color='blue', linestyle=':', linewidth=2, alpha=0.7,
                            label=f'均值: {mean_residual:.2f}')

                ax3.set_xlabel('训练集日期 (11月天数)', fontsize=12)
                ax3.set_ylabel('残差值', fontsize=12)
                ax3.set_title('ARIMA模型残差分布', fontsize=14, fontweight='bold')
                ax3.legend(fontsize=11)
                ax3.grid(True, alpha=0.3)
                ax3.set_xticks(train_dates)

                # 统计信息框
                stats_text = (f'均值: {residuals_train.mean():.2f}\n'
                              f'标准差: {residuals_train.std():.2f}\n'
                              f'最大值: {residuals_train.max():.2f}\n'
                              f'最小值: {residuals_train.min():.2f}')

                ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=11,
                         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                                                            facecolor="lightgray", alpha=0.7))

                figs['residual_analysis'] = fig3

            # 4. 特征重要性图
            fig4, ax4 = plt.subplots(figsize=(12, 8))

            if 'feature_importance' in self.results:
                feature_importance = self.results['feature_importance'].head(10)

                # 特征名称映射
                feature_names_map = {
                    'day': '日期', 'day_of_week': '星期', 'day_of_month': '月内天数',
                    'is_weekend': '是否周末', 'is_month_end': '是否月末',
                    'order_count': '订单数', 'avg_sale_amount': '平均销售额',
                    'total_sale': '总销售额', 'gross_profit_margin': '毛利率',
                    'customer_value': '单客价值', 'female_ratio': '女性比例',
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
                ax4.set_title('XGBoost特征重要性排名', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3, axis='x')

                # 添加重要性数值标签
                for i, v in enumerate(feature_importance['importance']):
                    ax4.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')

                figs['feature_importance'] = fig4

            self.results['visualizations'] = figs
            return True

        except Exception as e:
            st.error(f"可视化生成错误: {str(e)}")
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
                f"可视化图表生成完成：{len(self.results['visualizations'])}个分析图表"
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

    def abc_analysis(self):
        """ABC分类分析"""
        try:
            # 1. 按商品品类ABC分类
            if all(x in self.df.columns for x in ['商品品类', '销售额', '利润']):
                category_metrics = self.df.groupby('商品品类').agg({
                    '销售额': 'sum',
                    '利润': 'sum',
                    '销售数': 'count'
                }).reset_index()
                category_metrics = category_metrics.sort_values('销售额', ascending=False)
                category_metrics['销售额累计占比%'] = (
                        category_metrics['销售额'].cumsum() / category_metrics['销售额'].sum() * 100).round(2)
                category_metrics['利润累计占比%'] = (
                        category_metrics['利润'].cumsum() / category_metrics['利润'].sum() * 100).round(2)

                # ABC分类规则
                def assign_abc(cumulative_percent):
                    if cumulative_percent <= 70:
                        return 'A类（核心）'
                    elif cumulative_percent <= 90:
                        return 'B类（潜力）'
                    else:
                        return 'C类（长尾）'

                category_metrics['ABC分类（按销售额）'] = category_metrics['销售额累计占比%'].apply(assign_abc)
                category_metrics['ABC分类（按利润）'] = category_metrics['利润累计占比%'].apply(assign_abc)

                # 2. 按区域ABC分类
                if '区域' in self.df.columns:
                    region_metrics = self.df.groupby('区域').agg({
                        '销售额': 'sum',
                        '利润': 'sum'
                    }).reset_index()
                    region_metrics = region_metrics.sort_values('销售额', ascending=False)
                    region_metrics['销售额累计占比%'] = (
                            region_metrics['销售额'].cumsum() / region_metrics['销售额'].sum() * 100).round(2)
                    region_metrics['ABC分类（按销售额）'] = region_metrics['销售额累计占比%'].apply(assign_abc)

                    self.results['region_abc'] = region_metrics

                self.results['category_abc'] = category_metrics
                return True
            st.warning("缺少ABC分类所需字段（商品品类、销售额、利润）")
            return False
        except Exception as e:
            st.error(f"ABC分类错误: {str(e)}")
            return False

    def price_sensitivity_analysis(self):
        """价格敏感度分析"""
        try:
            if not all(x in self.df.columns for x in ['商品品类', '实际售价', '销售数']):
                st.warning("缺少价格敏感度分析所需字段（商品品类、实际售价、销售数）")
                return False

            sensitivity_results = []
            # 1. 按品类分析价格敏感度
            for category in self.df['商品品类'].unique():
                category_data = self.df[self.df['商品品类'] == category].copy()
                if len(category_data) < 10:
                    st.info(f"商品品类【{category}】样本量不足10条，跳过分析")
                    continue

                # 等频8区间划分
                category_data['价格区间'] = pd.qcut(
                    category_data['实际售价'],
                    q=8,
                    labels=[f'区间{i}' for i in range(1, 9)],
                    duplicates='drop'
                )
                price_sales = category_data.groupby('价格区间').agg({
                    '实际售价': 'mean',
                    '销售数': 'sum'
                }).reset_index()

                # 计算敏感度系数
                slope, intercept, r_value, p_value, std_err = linregress(
                    price_sales['实际售价'],
                    price_sales['销售数']
                )
                sensitivity_coeff = slope / (price_sales['销售数'].mean() / price_sales['实际售价'].mean())

                # 敏感度等级判定
                if sensitivity_coeff < -0.3:
                    level = '高敏感度（价格主导）'
                elif sensitivity_coeff < -0.1:
                    level = '中敏感度（价格+品质）'
                else:
                    level = '低敏感度（品质主导）'

                sensitivity_results.append({
                    '分析维度': '商品品类',
                    '维度值': category,
                    '价格弹性系数': sensitivity_coeff.round(4),
                    'R²（拟合优度）': round(r_value ** 2, 4),
                    '敏感度等级': level,
                    '样本量': len(category_data)
                })

            # 2. 按人群分析价格敏感度
            if '客户性别' in self.df.columns:
                st.info("开始按客户性别分析价格敏感度")
                for gender in self.df['客户性别'].unique():
                    gender_data = self.df[self.df['客户性别'] == gender].copy()
                    if len(gender_data) < 20:
                        st.info(f"客户性别【{gender}】样本量不足20条，跳过分析")
                        continue

                    gender_data['价格区间'] = pd.qcut(
                        gender_data['实际售价'],
                        q=8,
                        labels=[f'区间{i}' for i in range(1, 9)],
                        duplicates='drop'
                    )
                    price_sales = gender_data.groupby('价格区间').agg({
                        '实际售价': 'mean',
                        '销售数': 'sum'
                    }).reset_index()

                    slope, intercept, r_value, p_value, std_err = linregress(
                        price_sales['实际售价'],
                        price_sales['销售数']
                    )
                    sensitivity_coeff = slope / (price_sales['销售数'].mean() / price_sales['实际售价'].mean())

                    level = '高敏感度' if sensitivity_coeff < -0.3 else '中敏感度' if sensitivity_coeff < -0.1 else '低敏感度'
                    sensitivity_results.append({
                        '分析维度': '客户性别',
                        '维度值': gender,
                        '价格弹性系数': sensitivity_coeff.round(4),
                        'R²（拟合优度）': round(r_value ** 2, 4),
                        '敏感度等级': level,
                        '样本量': len(gender_data)
                    })

            sensitivity_df = pd.DataFrame(sensitivity_results)
            self.results['price_sensitivity'] = sensitivity_df
            st.success("价格敏感度分析完成")

            # 可视化：高敏感度品类TOP5
            high_sensitivity = sensitivity_df[sensitivity_df['分析维度'] == '商品品类'].nsmallest(5, '价格弹性系数')
            if len(high_sensitivity) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='维度值', y='价格弹性系数', data=high_sensitivity, palette='Reds')
                ax.set_title('商品品类价格敏感度TOP5（弹性系数越小越敏感）', fontsize=14)
                ax.set_xlabel('商品品类', fontsize=12)
                ax.set_ylabel('价格弹性系数', fontsize=12)
                ax.axhline(y=-0.3, color='red', linestyle='--', alpha=0.7, label='高敏感度阈值（-0.3）')
                ax.legend()
                plt.xticks(rotation=45)
                self.results['sensitivity_plot'] = fig
                st.pyplot(fig)

            return True
        except Exception as e:
            st.error(f"价格敏感度分析错误: {str(e)}")
            return False

    def generate_operation_strategy(self):
        """生成运营策略"""
        try:
            strategies = []
            # 1. 商品品类策略
            if 'category_abc' in self.results and 'price_sensitivity' in self.results:
                category_abc = self.results['category_abc']
                price_sensitivity = self.results['price_sensitivity'][
                    self.results['price_sensitivity']['分析维度'] == '商品品类']

                for _, abc_row in category_abc.iterrows():
                    category = abc_row['商品品类']
                    abc_sales = abc_row['ABC分类（按销售额）']
                    sens_row = price_sensitivity[price_sensitivity['维度值'] == category]
                    if len(sens_row) == 0:
                        continue
                    sens_level = sens_row['敏感度等级'].iloc[0]

                    if abc_sales == 'A类（核心）':
                        if sens_level == '高敏感度（价格主导）':
                            strategy = "保销量：日常定价维持品类均价-5%，大促期间'满减+赠品'"
                            inventory = "高安全库存（月销量1.5倍），提前30天备货"
                        else:
                            strategy = "提利润：高端款溢价10%-15%，常规款维持均价，非大促不降价"
                            inventory = "中等库存（月销量1.2倍），建立区域共享库存池"
                    elif abc_sales == 'B类（潜力）':
                        strategy = "促转化：组合促销，新用户首单折扣5%-8%，提升品类渗透率"
                        inventory = "动态库存（参考预测销量1.1倍），每月调整一次"
                    else:
                        strategy = "清库存：捆绑销售，或限时折扣30%-50%，减少资金占用"
                        inventory = "低库存（月销量0.8倍），滞销超60天直接下架"

                    strategies.append({
                        '策略维度': '商品品类',
                        '维度值': category,
                        'ABC分类': abc_sales,
                        '价格敏感度': sens_level,
                        '定价策略': strategy,
                        '库存策略': inventory,
                        '优先级': '高' if abc_sales == 'A类（核心）' else '中' if abc_sales == 'B类（潜力）' else '低'
                    })

            # 2. 区域策略
            if 'region_abc' in self.results:
                for _, region_row in self.results['region_abc'].iterrows():
                    region = region_row['区域']
                    abc_sales = region_row['ABC分类（按销售额）']

                    if abc_sales == 'A类（核心）':
                        strategy = "重点投入：增加区域专属促销，优化物流时效，提升用户留存"
                        resource = "优先配置仓储资源，增加客服团队"
                    elif abc_sales == 'B类（潜力）':
                        strategy = "渗透拓展：与区域KOL合作推广，开设线下体验点"
                        resource = "适度投入广告预算，测试用户偏好"
                    else:
                        strategy = "低成本覆盖：通过社区团购、下沉渠道触达"
                        resource = "控制成本，复用核心区域资源"

                    strategies.append({
                        '策略维度': '区域',
                        '维度值': region,
                        'ABC分类': abc_sales,
                        '运营策略': strategy,
                        '资源配置': resource,
                        '优先级': '高' if abc_sales == 'A类（核心）' else '中' if abc_sales == 'B类（潜力）' else '低'
                    })

            strategy_df = pd.DataFrame(strategies)
            self.results['operation_strategy'] = strategy_df
            return True
        except Exception as e:
            st.error(f"策略生成错误: {str(e)}")
            return False

    def generate_all_results(self):
        """生成所有优化结果"""
        try:
            abc_success = self.abc_analysis()
            sensitivity_success = self.price_sensitivity_analysis()
            strategy_success = self.generate_operation_strategy() if (abc_success and sensitivity_success) else False

            # 整理结果文件
            result_files = {}
            progress_log = []

            if abc_success:
                result_files['01_ABC分类结果（商品品类+区域）.xlsx'] = pd.ExcelWriter(io.BytesIO())
                with result_files['01_ABC分类结果（商品品类+区域）.xlsx'] as writer:
                    self.results['category_abc'].to_excel(writer, sheet_name='商品品类ABC', index=False)
                    if 'region_abc' in self.results:
                        self.results['region_abc'].to_excel(writer, sheet_name='区域ABC', index=False)
                result_files['01_ABC分类结果（商品品类+区域）.xlsx'] = result_files[
                    '01_ABC分类结果（商品品类+区域）.xlsx'].book
                progress_log.append("ABC分类完成：商品品类按销售额/利润分类，区域按销售额分类")

            if sensitivity_success:
                result_files['02_价格敏感度分析结果（品类+人群）.xlsx'] = self.results['price_sensitivity']
                progress_log.append(
                    f"价格敏感度分析完成：覆盖{len(self.results['price_sensitivity'])}个维度值")

            if strategy_success:
                result_files['03_运营策略推荐.xlsx'] = self.results['operation_strategy']
                progress_log.append(
                    f"运营策略生成完成：{len(self.results['operation_strategy'])}条策略")

            return result_files, progress_log
        except Exception as e:
            return None, [f"优化分析错误: {str(e)}"]


# ============================================================================
# 页面函数
# ============================================================================
def show_project_overview():
    """项目概览页面"""
    st.header("🎯 项目概览")

    st.markdown(
        '<div class="fix-note"><strong>系统功能：</strong><br>1. 支持导入任意电商Excel数据，按论文标准流程处理<br>2. 自动生成6个标准化输出文件<br>3. 提供完整的数据分析、建模和可视化功能</div>',
        unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
    ### 系统功能概述
    完整的电商销售分析流程：
    - **数据预处理**: 按论文要求生成6个标准化输出文件
    - **多维特征分析**: 品类×区域×利润热力图、客户-商品聚类
    - **销售预测**: ARIMA+XGBoost混合模型预测未来销售趋势
    - **运营优化**: ABC分类、价格敏感度分析、可落地的运营策略
    """)

    with col2:
        st.metric("标准输出文件", "6个")
        st.metric("分析任务", "4个")
        st.metric("支持数据格式", "Excel/CSV")

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

    if not st.session_state.task1_completed:
        st.warning("请先完成数据预处理（任务1）")
        return

    df = st.session_state.processed_data
    column_types = st.session_state.column_types

    # 分析模式选择
    analysis_mode = st.radio(
        "选择分析模式:",
        ["📊 Python可视化展示", "📁 论文图表数据导出"],
        horizontal=True
    )

    if st.button("🚀 执行多维特征分析", type="primary"):
        with st.spinner("正在执行多维分析..."):
            analyzer = EnhancedTask2Analyzer(df, column_types)

            # 执行基础分析（热力图和聚类）
            analyzer.create_heatmaps()
            analyzer.perform_clustering_analysis()

            # 生成所有分析数据
            all_analysis_data = analyzer.generate_all_analysis_data()

            st.session_state.task2_results = analyzer.results
            st.session_state.task2_analysis_data = all_analysis_data
            st.session_state.task2_completed = True

            st.success("✅ 多维特征分析完成！")

            if analysis_mode == "📊 Python可视化展示":
                show_python_visualizations(analyzer)
            else:
                show_data_export_interface(all_analysis_data)
    else:
        st.info("""
        **多维特征分析功能说明：**

        **📊 Python可视化展示模式：**
        - 交叉维度热力图分析
        - 客户-商品聚类分析
        - 系统内置可视化图表

        **📁 论文图表数据导出模式：**
        - 城市分布数据
        - 省份分布数据
        - 城市分级数据
        - 区域分级数据
        - 性别-品类数据
        - 年龄-性别数据
        - 时间序列数据
        - 相关性矩阵数据

        点击上方按钮开始分析！
        """)
def task3_sales_forecast():
    """任务3：销售预测页面"""
    st.header("📈 任务3: 销售预测")

    if not st.session_state.get('task1_completed', False):
        st.warning("请先完成数据预处理（任务1）")
        return

    # 获取预处理后的数据
    df = st.session_state.processed_data
    column_types = st.session_state.column_types

    # 执行预测
    if st.button("🚀 执行ARIMA-XGBoost混合预测", type="primary"):
        with st.spinner("预测中...（使用ARIMA(2,1,2)+XGBoost混合模型）"):
            forecaster = Task3Forecaster(df, column_types)
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
                    st.markdown("#### 3. ARIMA模型残差分析图")
                    st.pyplot(viz_results['residual_analysis'])
                    st.markdown("""
                    **图表说明：**
                    - 显示ARIMA模型在训练集上的残差分布
                    - 残差越接近0且波动越小，说明ARIMA模型拟合越好
                    - 为XGBoost提供学习目标
                    """)

                    # 特征重要性排名图
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
        - 使用ARIMA(2,1,2)模型捕捉时间序列趋势
        - 使用XGBoost模型学习ARIMA的残差模式
        - 最终预测 = ARIMA预测 + XGBoost残差预测
        - 测试集：11月25-30日（后6天数据）
        """)

def task4_operation_optimization():
    """任务4：运营优化页面"""
    st.header("💡 任务4: 运营策略优化")

    if not st.session_state.task1_completed:
        st.warning("请先完成数据预处理（任务1）")
        return

    # 获取预处理后的数据
    df = st.session_state.processed_data
    column_types = st.session_state.column_types

    # 执行运营优化分析
    if st.button("🚀 执行运营优化分析", type="primary"):
        with st.spinner("分析中...（生成可落地策略）"):
            optimizer = Task4Optimizer(df, column_types)
            result_files, progress_log = optimizer.generate_all_results()

            if result_files:
                st.session_state.task4_results = optimizer.results
                st.session_state.task4_completed = True
                st.success("✅ 运营优化分析完成！")

                # 1. 展示ABC分类结果
                st.subheader("1. ABC分类结果（商品品类+区域）")
                if 'category_abc' in optimizer.results:
                    st.subheader("1.1 商品品类ABC分类（按销售额/利润）")
                    category_abc = optimizer.results['category_abc']
                    st.dataframe(category_abc[['商品品类', '销售额', '利润', '销售额累计占比%',
                                               'ABC分类（按销售额）']].round(2))

                if 'region_abc' in optimizer.results:
                    st.subheader("1.2 区域ABC分类（按销售额）")
                    region_abc = optimizer.results['region_abc']
                    st.dataframe(region_abc[['区域', '销售额', '利润', 'ABC分类（按销售额）']].round(2))

                # 2. 展示价格敏感度分析结果
                st.subheader("2. 价格敏感度分析（品类+人群）")
                if 'price_sensitivity' in optimizer.results:
                    sensitivity_df = optimizer.results['price_sensitivity']
                    # 分维度展示
                    tab1, tab2 = st.tabs(["商品品类敏感度", "客户人群敏感度"])

                    with tab1:
                        cat_sens = sensitivity_df[sensitivity_df['分析维度'] == '商品品类']
                        cat_sens_sorted = cat_sens.sort_values('价格弹性系数')
                        st.dataframe(cat_sens_sorted[['维度值', '价格弹性系数', 'R²（拟合优度）', '敏感度等级',
                                                      '样本量']].round(4))

                    with tab2:
                        people_sens = sensitivity_df[sensitivity_df['分析维度'] == '客户性别']
                        if len(people_sens) > 0:
                            st.dataframe(
                                people_sens[['维度值', '价格弹性系数', '敏感度等级', '样本量']].round(4))
                        else:
                            st.info("暂无客户人群敏感度数据")

                # 3. 展示运营策略推荐
                st.subheader("3. 可落地运营策略推荐")
                if 'operation_strategy' in optimizer.results:
                    strategy_df = optimizer.results['operation_strategy']
                    # 按优先级筛选展示
                    tab_high, tab_mid, tab_low = st.tabs(
                        ["高优先级策略（A类核心）", "中优先级策略（B类潜力）", "低优先级策略（C类长尾）"])

                    with tab_high:
                        high_strategy = strategy_df[strategy_df['优先级'] == '高']
                        if len(high_strategy) > 0:
                            st.dataframe(high_strategy)

                    with tab_mid:
                        mid_strategy = strategy_df[strategy_df['优先级'] == '中']
                        if len(mid_strategy) > 0:
                            st.dataframe(mid_strategy)

                    with tab_low:
                        low_strategy = strategy_df[strategy_df['优先级'] == '低']
                        if len(low_strategy) > 0:
                            st.dataframe(low_strategy)

                # 4. 文件下载
                st.subheader("📥 运营优化结果文件下载")
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

                # 5. 分析日志
                st.subheader("4. 分析日志")
                for log in progress_log:
                    st.write(f"▪️ {log}")


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
# 主应用函数
# ============================================================================
# ============================================================================
# 主应用函数 - 修复路由问题
# ============================================================================
def main():
    """主应用函数"""
    # 页面配置 - 在这里设置一次
    st.set_page_config(
        page_title="电商销售分析与策略优化系统",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown('<div class="main-header">📊 电商销售分析与策略优化系统</div>',
                unsafe_allow_html=True)
    st.markdown(
        f"### 当前文件：{st.session_state.get('current_file', '未上传文件')}")

    # 侧边栏导航
    with st.sidebar:
        st.title("导航菜单")
        selected_task = st.radio(
            "选择分析任务:",
            [
                "项目概览",
                "数据预处理",
                "多维销售特征分析",
                "销售预测分析",
                "运营策略优化",
                "系统状态"
            ]
        )

        # 任务状态概览
        st.markdown("---")
        st.subheader("任务完成状态")
        tasks_status = [
            ("数据预处理", st.session_state.task1_completed),
            ("多维特征分析", st.session_state.task2_completed),
            ("销售预测", st.session_state.task3_completed),
            ("运营优化", st.session_state.task4_completed)
        ]
        for task_name, completed in tasks_status:
            status_class = "status-completed" if completed else "status-pending"
            icon = "✅" if completed else "⏳"
            st.markdown(f'<div class="{status_class}">{icon} {task_name}</div>', unsafe_allow_html=True)

    # 页面路由
    if selected_task == "项目概览":
        show_project_overview()
    elif selected_task == "数据预处理":
        task1_data_preprocessing()
    elif selected_task == "多维销售特征分析":
        enhanced_task2_multidimensional_analysis()
    elif selected_task == "销售预测分析":
        task3_sales_forecast()
    elif selected_task == "运营策略优化":
        task4_operation_optimization()
    elif selected_task == "系统状态":
        show_system_status()

    # 页脚
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "电商销售分析与策略优化系统 | 支持论文要求的标准化输出"
        "</div>",
        unsafe_allow_html=True
    )

# ============================================================================
# 运行应用
# ============================================================================
if __name__ == "__main__":
    main()
