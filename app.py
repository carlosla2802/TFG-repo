import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide")  # Configuración de la página para un layout más ancho

# Carga de datos
df = pd.read_csv("final_dataset.csv")

age_mapping = {6.0: '45-49 years', 7.0: '50-54 years', 8.0: '55-59 years', 9.0: '60-64 years', 10.0: '65-69 years', 11.0: '70-74 years', 12.0: '75-79 years', 13.0: '80 years or more'}

df['_AGEG5YR'] = df['_AGEG5YR'].map(age_mapping)

df_sorted = df.sort_values('_AGEG5YR')

# Function to check it its real intenger (number.0 and not number.x)
def is_integer(x):
    try:
        return float(x).is_integer()
    except ValueError:
        return False

# Finds columns that are of type 'float64' or 'int64' and have 15 unique values or less
categories = [col for col in df.columns if (df[col].dtype == 'float64' or df[col].dtype == 'int64') and df[col].nunique() <= 15]

# Convert these columns to type 'category' only if all values are integers
for col in categories:
    # Checks if all values in the column are actually integers (.0 as fractional part)
    if all(df[col].dropna().apply(is_integer)):
        df[col] = df[col].astype('category')



# Inicialización de la app de Streamlit
st.title('📊  Features Visualizer by Age')

# Seleccionar la columna para visualización
column = st.selectbox('Select the feature to display:', df.columns, index=df.columns.get_loc('CVDSTRK3'))


if df[column].dtype == 'category':
    # Crear el gráfico de líneas principal con Plotly
    line_data = df.groupby([column, '_AGEG5YR']).size().reset_index(name='Frequency')
    fig_line = px.line(line_data, x='_AGEG5YR', y='Frequency', color=column, labels={'_AGEG5YR': 'Age Group', 'Frequency': 'Frequency'},
                    title=f'Distribution of {column} by Age Group')
    fig_line.update_traces(mode='lines+markers')
    st.plotly_chart(fig_line, use_container_width=True)
else:
    fig_box = px.box(df, x='_AGEG5YR', y=column, labels={'_AGEG5YR': 'Age Group', column: 'Value'},
                    title=f'Distribution of {column} by Age Group')
    st.plotly_chart(fig_box, use_container_width=True)

# Crear subplots para cada grupo de edad con condiciones específicas
fig_bar = go.Figure()
age_groups = sorted(df['_AGEG5YR'].unique())

for age in age_groups:
    age_data = df_sorted[df_sorted['_AGEG5YR'] == age]
    # Asumimos que solo hay dos condiciones, 0 y 1
    for cond in [0, 1]:  # Asegurar que la condición 0 está a la izquierda y 1 a la derecha
        subset = age_data[age_data['_MICHD'] == cond]
        fig_bar.add_trace(go.Bar(
            x=[f'Age {age}'], 
            y=[subset[column].count()],
            name=f'MICHD {cond}',
            offsetgroup=cond,  # Usar la condición como offsetgroup
            base=None  # base set to None for normal stacking
        ))

# Ajustar layout del gráfico de barras y agregar anotaciones
annotations = []
for i, age in enumerate(age_groups):
    # Ajustar la posición vertical de las anotaciones para estar bajo las barras
    annotations.append(dict(
        x=i - 0.2,  # Ajustar posición para la barra izquierda
        y=-300,  # Posición vertical ajustada
        text='MICHD 0',
        showarrow=False,
        xref="x",
        yref="y",
        align="center"
    ))
    annotations.append(dict(
        x=i + 0.2,  # Ajustar posición para la barra derecha
        y=-300,  # Posición vertical ajustada
        text='MICHD 1',
        showarrow=False,
        xref="x",
        yref="y",
        align="center"
    ))

fig_bar.update_layout(
    barmode='group',
    title='Distribution by Age Group and MICHD',
    xaxis_title='Age Group',
    yaxis_title='Frequency',
    bargap=0.3,  # Ajustar el espacio entre grupos de barras
    showlegend=False,  # Ocultar la leyenda
    annotations=annotations
)

st.plotly_chart(fig_bar, use_container_width=True)

