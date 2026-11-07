import json
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title='Вино для подборки', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model(path, version):
    return joblib.load(path)


@st.cache_data
def load_test(path, version):
    return pd.read_csv(path)


@st.cache_data
def test_proba(_model, X, model_version):
    return _model.predict_proba(X)[:, 1]


def read_wines(content, columns):
    table = pd.read_csv(BytesIO(content))
    missing = [name for name in columns if name not in table.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if table.empty:
        raise ValueError('В файле нет строк с винами.')
    features = table[columns].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError('В признаках есть пропуски или бесконечные значения.')
    if not features['is_red'].isin([0, 1]).all():
        raise ValueError('В столбце is_red должны быть только 0 и 1.')
    return table, features


model_path = APP_DIR / 'model.joblib'
model_version = model_path.stat().st_mtime_ns
model = load_model(str(model_path), model_version)
test_path = APP_DIR / 'test.csv'
test = load_test(str(test_path), test_path.stat().st_mtime_ns)
config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))
feature_names = list(model.feature_names_in_)
X_test = test[feature_names]
y_test = test['target']
probabilities = test_proba(model, X_test, model_version)

st.title('Вино для подборки')
st.caption('Хорошее вино — экспертная оценка от 7 баллов. Модель использует лабораторные показатели.')
threshold = st.sidebar.slider(
    'Порог отбора', min_value=0.0, max_value=1.0,
    value=float(config['threshold']), step=0.001, format='%.3f', key='threshold'
)
st.sidebar.write('Порог применяется на всех вкладках.')
st.sidebar.caption(f'Порог из валидации: {config["threshold"]:.3f}')

single_tab, metrics_tab, batch_tab, weights_tab = st.tabs(
    ['Одно вино', 'Модельные метрики', 'Загрузить CSV', 'Что выучила модель']
)

labels = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
    'is_red': 'Красное вино'
}

with single_tab:
    st.subheader('Оценка одного вина')
    with st.form('wine_form'):
        inputs = {}
        left, right = st.columns(2)
        for i, name in enumerate(feature_names):
            with [left, right][i % 2]:
                if name == 'is_red':
                    inputs[name] = int(st.toggle('Красное вино', value=bool(X_test[name].median())))
                else:
                    step = 0.00001 if name == 'density' else 0.001
                    inputs[name] = st.number_input(
                        labels[name], min_value=float(X_test[name].min()),
                        max_value=float(X_test[name].max()), value=float(X_test[name].median()),
                        step=step, format='%.5f' if name == 'density' else '%.3f', key=name
                    )
        submitted = st.form_submit_button('Рассчитать')
    if submitted or 'wine_probability' not in st.session_state:
        row = pd.DataFrame([inputs], columns=feature_names)
        st.session_state['wine_probability'] = float(model.predict_proba(row)[0, 1])
    probability = st.session_state['wine_probability']
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    if probability >= threshold:
        st.success('Добавить в подборку')
    else:
        st.info('Пропустить')
    st.caption('После изменения показателей нажмите «Рассчитать». Значения и диапазоны взяты из test.csv.')

with metrics_tab:
    st.subheader('Качество на тестовой выборке')
    predictions = probabilities >= threshold
    precision = metrics.precision_score(y_test, predictions, zero_division=0)
    recall = metrics.recall_score(y_test, predictions, zero_division=0)
    f1 = metrics.f1_score(y_test, predictions, zero_division=0)
    for column, label, value in zip(st.columns(3), ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
        column.metric(label, f'{value:.3f}')
    st.write(
        'Если увеличить порог, в подборке останется меньше вин: обычно она станет точнее, '
        'но больше хороших вин будет пропущено. Если уменьшить порог, подборка расширится, '
        'а риск включить посредственное вино вырастет.'
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    cm = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics.ConfusionMatrixDisplay(cm, display_labels=['Остальные', 'Хорошие']).plot(ax=ax, cmap='Blues')
    ax.set(xlabel='Решение модели', ylabel='Настоящий класс')
    matrix_column, explanation_column = st.columns([1, 2])
    matrix_column.pyplot(fig)
    explanation_column.write(
        f'В подборку попадёт {int(predictions.sum())} вин. '
        f'Из них хороших: {int(cm[1, 1])}, остальных: {int(cm[0, 1])}. '
        f'Хороших вин пропущено: {int(cm[1, 0])}.'
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    metrics.RocCurveDisplay.from_predictions(y_test, probabilities, ax=axes[0], name='LR')
    tn, fp, fn, tp = cm.ravel()
    axes[0].plot([0, 1], [0, 1], '--', color='grey')
    axes[0].scatter(fp / (fp + tn), recall, color='red', s=65, label=f'Порог {threshold:.3f}', zorder=3)
    axes[0].set_title('ROC-кривая')
    axes[0].legend()
    metrics.PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=axes[1], name='LR')
    axes[1].axhline(y_test.mean(), ls='--', color='grey', label='Случайный отбор')
    axes[1].scatter(recall, precision, color='red', s=65, label=f'Порог {threshold:.3f}', zorder=3)
    axes[1].set_title('PR-кривая')
    axes[1].legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f'{len(test)} вин. ROC-AUC = {metrics.roc_auc_score(y_test, probabilities):.3f}; '
        f'PR-AUC (average precision) = {metrics.average_precision_score(y_test, probabilities):.3f}. '
        'Эта вкладка показывает чувствительность к порогу; исходный порог выбран на валидации.'
    )

with batch_tab:
    st.subheader('Пакетный отбор')
    st.write('Загрузите CSV с теми же признаками, что в примере. Столбец target не нужен.')
    st.download_button(
        'Скачать пример CSV', X_test.head(5).to_csv(index=False).encode('utf-8'),
        file_name='wine_example.csv', mime='text/csv'
    )
    uploaded = st.file_uploader('CSV с винами', type=['csv'])
    if uploaded is not None:
        try:
            table, features = read_wines(uploaded.getvalue(), feature_names)
            result = table.copy()
            result['probability'] = model.predict_proba(features)[:, 1]
            result['recommend'] = (result['probability'] >= threshold).astype(int)
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.download_button(
                'Скачать предсказания', result.to_csv(index=False).encode('utf-8-sig'),
                file_name='wine_predictions.csv', mime='text/csv'
            )
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f'Не удалось обработать CSV. {error}')

with weights_tab:
    st.subheader('Вклад признаков')
    coefficients = pd.Series(model[-1].coef_.ravel(), index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([labels[name] for name in coefficients.index], coefficients,
            color=np.where(coefficients >= 0, '#477c64', '#b3616e'))
    ax.axvline(0, color='grey', linewidth=1)
    ax.set_xlabel('Коэффициент при стандартизованном признаке')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    positive = ', '.join(labels[name].lower() for name in coefficients.nlargest(3).index)
    negative = ', '.join(labels[name].lower() for name in coefficients.nsmallest(3).index)
    st.write(f'Сильнее всего за хорошее вино: {positive}. Против: {negative}.')
    st.caption(
        'Коэффициент показывает изменение логарифма шансов при увеличении признака '
        'на одно стандартное отклонение, если остальные признаки фиксированы. '
        'Это связь в данных, а не доказательство причинного влияния.'
    )
