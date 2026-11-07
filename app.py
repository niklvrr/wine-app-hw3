import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent
st.set_page_config(page_title='Подбор вин', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, features):
    return _model.predict_proba(features)[:, 1]


model = load_model()
test = load_test()
features = list(model.feature_names_in_)
X_test = test[features]
y_test = test['target']
probabilities = test_proba(model, X_test)
config = json.loads((APP_DIR / 'config.json').read_text())

st.title('Подбор хороших вин')
threshold = st.sidebar.slider('Порог рекомендации', 0.0, 1.0,
                              float(config['threshold']), 0.01)
st.sidebar.caption('Более высокий порог делает подборку строже.')

single_tab, metrics_tab, batch_tab, coefficients_tab = st.tabs(
    ['Одно вино', 'Модельные метрики', 'Загрузить CSV', 'Коэффициенты'])

with single_tab:
    st.subheader('Характеристики вина')
    values = {}
    left, right = st.columns(2)
    for index, feature in enumerate(features):
        container = left if index % 2 == 0 else right
        if feature == 'is_red':
            values[feature] = int(container.toggle('Красное вино',
                                                   value=bool(test[feature].median())))
        else:
            values[feature] = container.number_input(
                feature, min_value=float(test[feature].min()),
                max_value=float(test[feature].max()),
                value=float(test[feature].median()),
                format='%.4f')
    single = pd.DataFrame([values], columns=features)
    probability = float(model.predict_proba(single)[0, 1])
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    st.write('Рекомендовать' if probability >= threshold else 'Не рекомендовать')

with metrics_tab:
    predictions = (probabilities >= threshold).astype(int)
    precision = metrics.precision_score(y_test, predictions, zero_division=0)
    recall = metrics.recall_score(y_test, predictions, zero_division=0)
    f1 = metrics.f1_score(y_test, predictions, zero_division=0)
    col1, col2, col3 = st.columns(3)
    col1.metric('Precision', f'{precision:.3f}')
    col2.metric('Recall', f'{recall:.3f}')
    col3.metric('F1', f'{f1:.3f}')
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax_cm,
                                                    cmap='Blues', colorbar=False)
    st.pyplot(fig_cm)
    plt.close(fig_cm)
    fpr, tpr, _ = metrics.roc_curve(y_test, probabilities)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, probabilities)
    current_fpr = np.mean(predictions[y_test.to_numpy() == 0])
    current_tpr = recall
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr)
    axes[0].scatter(current_fpr, current_tpr, color='crimson', s=60)
    axes[0].set(xlabel='FPR', ylabel='TPR', title='ROC')
    axes[1].plot(pr_recall, pr_precision)
    axes[1].scatter(recall, precision, color='crimson', s=60)
    axes[1].axhline(y_test.mean(), ls='--', color='gray')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='PR')
    st.pyplot(fig)
    plt.close(fig)
    st.caption('При повышении порога вин в подборке становится меньше: обычно растёт precision и падает recall. При снижении порога вин становится больше, включая больше посредственных.')

with batch_tab:
    uploaded = st.file_uploader('CSV с 12 признаками, без target', type='csv')
    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            missing = [feature for feature in features if feature not in incoming.columns]
            if missing:
                st.error('Не хватает столбцов: ' + ', '.join(missing))
            else:
                batch = incoming[features].copy()
                for feature in features:
                    batch[feature] = pd.to_numeric(batch[feature], errors='raise')
                if batch.isna().any().any():
                    st.error('В признаках есть пустые значения. Заполните их и загрузите файл снова.')
                else:
                    result = incoming.copy()
                    result['probability_good'] = model.predict_proba(batch)[:, 1]
                    result['recommend'] = (result['probability_good'] >= threshold).astype(int)
                    st.dataframe(result)
                    st.download_button('Скачать прогнозы', result.to_csv(index=False).encode('utf-8'),
                                       file_name='wine_predictions.csv', mime='text/csv')
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(f'Не удалось прочитать признаки: {exc}')

with coefficients_tab:
    coefficients = pd.Series(model.named_steps['logisticregression'].coef_[0],
                             index=features).sort_values()
    fig_coef, ax_coef = plt.subplots(figsize=(9, 6))
    coefficients.plot.barh(ax=ax_coef, color=np.where(coefficients > 0, 'tab:green', 'tab:red'))
    ax_coef.set_xlabel('Коэффициент после StandardScaler')
    st.pyplot(fig_coef)
    plt.close(fig_coef)
    st.write(f'Сильнее всего за хорошее вино: {coefficients.idxmax()}. '
             f'Сильнее всего против: {coefficients.idxmin()}. '
             'Знак коэффициента описывает связь при прочих равных, а не причинный эффект.')
