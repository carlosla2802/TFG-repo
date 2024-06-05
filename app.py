import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np

st.set_page_config(layout="wide")  # Configuración de la página para un layout más ancho

# Carga de datos
df = pd.read_csv("final_dataset.csv")

# Inicialización de la app de Streamlit
st.title('📊  EDA Visualizer')

# Menú de selección en la barra lateral
option = st.sidebar.selectbox(
    'Select the visualization:',
    ('Feature by Age', 'Correlation Matrix', 'Correlations with MICHD', 'Univariate Analysis of Features', 'Bivariate Analysis of Features with _MICHD')
)

if option == 'Feature by Age':
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


    # Código para visualización de características por edad
    # ...
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



elif option == 'Correlation Matrix':
    # Cálculo de la matriz de correlación
    matriz_correlacion = df.corr()

    # Crear una máscara para ocultar la parte triangular superior excluyendo la diagonal
    mask = np.triu(np.ones_like(matriz_correlacion, dtype=bool))

    # Haciendo la matriz de correlación más manipulable para Plotly
    masked_corr = matriz_correlacion.mask(mask)

    # Crear el heatmap interactivo utilizando Plotly
    fig = px.imshow(masked_corr,
                    text_auto=True,
                    aspect='equal',  # Cambiado a 'equal' para mantener las proporciones cuadradas
                    labels=dict(color="Correlation"),
                    color_continuous_scale='RdBu_r',  # Cambio aquí para invertir la escala
                    zmin=-1, zmax=1)

    # Actualizar títulos y etiquetas
    fig.update_layout(
        title='Correlation Matrix',
        xaxis_title='Variables',
        yaxis_title='Variables',
        xaxis={'side': 'bottom'},  # Asegura que los labels del eje X estén en la parte inferior
        yaxis_tickmode='array',
        yaxis_tickvals=np.arange(len(df.columns)),
        yaxis_ticktext=df.columns,
        autosize=False,  # Permite un tamaño personalizado
        width=800,  # Anchura personalizable
        height=800  # Altura personalizable para hacerlo más cuadrado
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)


elif option == 'Correlations with MICHD':


    # Calcular la correlación específica con '_MICHD'
    df_sin_michd = df.drop('_MICHD', axis=1)
    corr = df_sin_michd.corrwith(df['_MICHD']).sort_values(ascending=False).to_frame()
    corr.columns = ['Correlations']

    # Definir los colores personalizados como una escala
    colors = ["#FFA07A", "#DC143C", "#B22222", "#E94B3C", "#2D2926"]  # Colores en formato HEX

    # Crear el gráfico de barras horizontal
    fig = px.bar(corr, y=corr.index, x='Correlations', orientation='h',
                title='Correlation with _MICHD',
                color='Correlations',
                color_continuous_scale=colors)

    # Actualizar el layout para mejorar la visualización
    fig.update_layout(
        coloraxis_colorbar=dict(
            title='Correlation',
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1', '-0.5', '0', '0.5', '1']
        ),
        xaxis_title='Correlation Value',
        yaxis_title='Variables',
        autosize=False,  # Permite un tamaño personalizado
        height=len(corr) * 40 + 100,  # Altura basada en el número de variables más algo de espacio adicional
        width=800  # Anchura fija
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)


elif option == 'Univariate Analysis of Features':

    # Lista de características a analizar
    final_features = df.columns  # Ajusta esto si solo quieres ciertas columnas

    # Crear subplots con 1 fila por cada par de características
    fig = make_subplots(rows=len(final_features)//2 + len(final_features)%2, cols=2, subplot_titles=[col for col in final_features])

    # Paleta de colores para alternar y evitar repetición en la misma fila
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Función para añadir trazas al gráfico
    def add_trace(column, row, col):
        unique_values = sorted(df[column].dropna().unique())
        if pd.api.types.is_numeric_dtype(df[column]):
            # Si la columna es numérica, crea un histograma
            fig.add_trace(go.Histogram(x=df[column], nbinsx=30, name=column), row=row, col=col)
        else:
            # Si la columna es categórica, crea un gráfico de barras
            counts = df[column].value_counts()
            fig.add_trace(go.Bar(x=counts.index, y=counts.values, name=column, marker_color=color_palette[(row-1)*2 + (col-1) % len(color_palette)]), row=row, col=col)
            if len(unique_values) <= 10:  # Ajuste para variables con pocas categorías
                fig.update_xaxes(row=row, col=col, categoryorder='array', categoryarray=unique_values)

    # Añadir gráficos a cada subplot
    for i, column in enumerate(final_features):
        add_trace(column, i // 2 + 1, i % 2 + 1)

    # Actualizar el layout para mejorar la visualización
    fig.update_layout(
        height=300 * (len(final_features)//2 + len(final_features)%2),  # Altura basada en el número de filas necesarias
        width=1200,  # Ancho suficiente para dos columnas
        title_text="Univariate Analysis of Features",
        showlegend=True
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)

elif option == 'Bivariate Analysis of Features with MICHD':
    # Código para la visualización de barras por edad y MICHD, como en tu código
    # ...
    pass