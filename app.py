import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
from math import ceil

st.set_page_config(layout="wide")  # Page configuration for a wider layout

# Sidebar and container styles
page_bg_img = """
<style>
[data-testid="stSidebar"]{
background-color: #dddddd;
opacity: 0.8;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

# Initialization of the Streamlit App
st.title('📊  EDA Visualizer')

# List of display options
visualization_options = ['Feature by Age', 'Correlation Matrix', 'Correlations with MICHD', 'Univariate Analysis', 'Bivariate Analysis']

# Initialization of the status variable if it does not exist or at restarting
if 'selected_option' not in st.session_state or st.session_state.selected_option not in visualization_options:
    st.session_state.selected_option = visualization_options[0]  # Select the first option by default

# Function to manage button clicks
def handle_button_click(option):
    st.session_state.selected_option = option

# Selection menu in the sidebar using buttons instead of selectbox
st.sidebar.markdown("**Select the visualization:**")
for option in visualization_options:
    if st.sidebar.button(option):
        handle_button_click(option)

st.sidebar.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Empty space to push the image to the bottom of the sidebar
for _ in range(10):
    st.sidebar.write("")

st.sidebar.image('Logo_uab.png', use_column_width=True, caption='Carlos Leta, Data Engineering')


# -----------------------------------------

# Data upload
df = pd.read_csv("final_dataset.csv")

if st.session_state.selected_option == 'Feature by Age':
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


    # Select column for display
    column = st.selectbox('Select the feature to display:', df.columns, index=df.columns.get_loc('CVDSTRK3'))


    if df[column].dtype == 'category':
        # Creating the main line chart with Plotly
        line_data = df.groupby([column, '_AGEG5YR']).size().reset_index(name='Frequency')
        fig_line = px.line(line_data, x='_AGEG5YR', y='Frequency', color=column, labels={'_AGEG5YR': 'Age Group', 'Frequency': 'Frequency'},
                        title=f'Distribution of {column} by Age Group')
        fig_line.update_traces(mode='lines+markers')
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        fig_box = px.box(df, x='_AGEG5YR', y=column, labels={'_AGEG5YR': 'Age Group', column: 'Value'},
                        title=f'Distribution of {column} by Age Group')
        st.plotly_chart(fig_box, use_container_width=True)

    # Create subplots for each age group with specific conditions
    fig_bar = go.Figure()
    age_groups = sorted(df['_AGEG5YR'].unique())

    for age in age_groups:
        age_data = df_sorted[df_sorted['_AGEG5YR'] == age]
        for cond in [0, 1]:  # Ensure that condition 0 is on the left and 1 on the right
            subset = age_data[age_data['_MICHD'] == cond]
            fig_bar.add_trace(go.Bar(
                x=[f'Age {age}'], 
                y=[subset[column].count()],
                name=f'MICHD {cond}',
                offsetgroup=cond,  # Use condition as offsetgroup
                base=None  # base set to None for normal stacking
            ))

    # Adjust bar chart layout and add annotations
    annotations = []
    for i, age in enumerate(age_groups):
        # Adjust the vertical position of the annotations to be under the bars.
        annotations.append(dict(
            x=i - 0.2,  # Adjust position for left bar
            y=-300,  # Vertical position adjusted
            text='MICHD 0',
            showarrow=False,
            xref="x",
            yref="y",
            align="center"
        ))
        annotations.append(dict(
            x=i + 0.2,  # Adjust position for the right bar
            y=-300,  # Vertical position adjusted
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
        bargap=0.3,  # Adjust the spacing between groups of busbars
        showlegend=False,  # Hide the legend
        annotations=annotations
    )

    st.plotly_chart(fig_bar, use_container_width=True)



elif st.session_state.selected_option == 'Correlation Matrix':
    # Calculation of the correlation matrix
    matriz_correlacion = df.corr()

    # Create a mask to hide the upper triangular part excluding the diagonal.
    mask = np.triu(np.ones_like(matriz_correlacion, dtype=bool))

    # Making the correlation matrix more manipulable for Plotly
    masked_corr = matriz_correlacion.mask(mask)

    # Create the interactive heatmap using Plotly
    fig = px.imshow(masked_corr,
                    text_auto=True,
                    aspect='equal',  # Changed to 'equal' to maintain square proportions
                    labels=dict(color="Correlation"),
                    color_continuous_scale='RdBu_r',  # Change here to invert the scale
                    zmin=-1, zmax=1)

    # Update titles and labels
    fig.update_layout(
        title='Correlation Matrix',
        xaxis_title='Variables',
        yaxis_title='Variables',
        xaxis={'side': 'bottom'},   # Make sure that the X-axis labels are at the bottom of the X axis.
        yaxis_tickmode='array',
        yaxis_tickvals=np.arange(len(df.columns)),
        yaxis_ticktext=df.columns,
        autosize=False,  # Allows custom sizing
        width=800,  # Customizable width
        height=800  # Customizable height to make it more square
    )

    # Display the graph in Streamlit
    st.plotly_chart(fig, use_container_width=True)


elif st.session_state.selected_option == 'Correlations with MICHD':


    # Calculate the specific correlation with '_MICHD'.
    df_sin_michd = df.drop('_MICHD', axis=1)
    corr = df_sin_michd.corrwith(df['_MICHD']).sort_values(ascending=False).to_frame()
    corr.columns = ['Correlations']

    # Define custom colors as a scale
    colors = ["#FFA07A", "#DC143C", "#B22222", "#E94B3C", "#2D2926"]  # Colores en formato HEX

    # Create the horizontal bar chart
    fig = px.bar(corr, y=corr.index, x='Correlations', orientation='h',
                title='Correlation with _MICHD',
                color='Correlations',
                color_continuous_scale=colors)

    # Update the layout to improve visualization
    fig.update_layout(
        coloraxis_colorbar=dict(
            title='Correlation',
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1', '-0.5', '0', '0.5', '1']
        ),
        xaxis_title='Correlation Value',
        yaxis_title='Variables',
        autosize=False,  # Allows custom sizing
        height=len(corr) * 40 + 100,  # Height based on the number of variables plus some additional space
        width=800  # Fixed width
    )

    # Display the graph in Streamlit
    st.plotly_chart(fig, use_container_width=True)


elif st.session_state.selected_option == 'Univariate Analysis':
    # Defines the layout of the graphics
    rows = ceil(len(df.columns) / 2)  # Adjusts according to the number of graphs
    cols = 2
    # fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'Bar Chart of {col}' if df[col].dtype == 'O' else f'Histogram of {col}' for col in df.columns])
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'{col}' for col in df.columns])


    # Fill in the subcharts
    row = 1
    col = 1
    for i, column in enumerate(df.columns):
        if df[column].dtype == np.number and len(df[column].unique()) == 2:  # Checks if the column is numeric and binaryia
            # Uses a bar chart for binary variables
            counts = df[column].value_counts()
            fig.add_trace(go.Bar(x=[0, 1], y=[counts.get(0, 0), counts.get(1, 0)], width=0.4), row=row, col=col)
            fig.update_xaxes(tickvals=[0, 1], row=row, col=col)
        elif df[column].dtype == np.number:
            fig.add_trace(go.Histogram(x=df[column], nbinsx=30), row=row, col=col)
        else:
            counts = df[column].value_counts()
            fig.add_trace(go.Bar(x=counts.index, y=counts.values), row=row, col=col)

        col += 1
        if col > cols:
            col = 1
            row += 1

    # Update the design
    fig.update_layout(height=300 * rows, width=1050, title_text="Univariate Analysis of Features", showlegend=False)


    # Streamlit integration
    st.plotly_chart(fig)

elif st.session_state.selected_option == 'Bivariate Analysis':

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

    #--------------------------------------------------


    # Determines the number of rows required for two columns
    rows = ceil(len(df.columns) / 2)
    cols = 2

    # Create a figure to contain the subgraphs with two columns
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'Bar Chart of {col} by _MICHD' if df[col].dtype == np.number and df[col].nunique() == 2 else f'Violin Plot of {col} by _MICHD' if pd.api.types.is_numeric_dtype(df[col]) else f'Count Plot of {col} by _MICHD' for col in df.columns if col != '_MICHD'])

    # Variables to control the current position of the subplot
    current_row = 1
    current_col = 1

    for i, column in enumerate(df.columns):
        if column != '_MICHD':
            if df[column].dtype == np.number and df[column].nunique() == 2:  # Check if the column is numeric and binary
                values_0 = df[df['_MICHD'] == 0][column].value_counts().sort_index()
                values_1 = df[df['_MICHD'] == 1][column].value_counts().sort_index()
                fig.add_trace(go.Bar(x=[0, 1], y=[values_0.get(0, 0), values_0.get(1, 0)], name=f'{column} (_MICHD 0)', marker_color='green'), row=current_row, col=current_col)
                fig.add_trace(go.Bar(x=[0, 1], y=[values_1.get(0, 0), values_1.get(1, 0)], name=f'{column} (_MICHD 1)', marker_color='red'), row=current_row, col=current_col)
            elif pd.api.types.is_numeric_dtype(df[column]):
                fig.add_trace(go.Violin(y=df[df['_MICHD']==0][column], name=f'{column} (_MICHD 0)', line_color='green'), row=current_row, col=current_col)
                fig.add_trace(go.Violin(y=df[df['_MICHD']==1][column], name=f'{column} (_MICHD 1)', line_color='red'), row=current_row, col=current_col)
            else:
                values_0 = df[df['_MICHD'] == 0][column].value_counts().sort_index()
                values_1 = df[df['_MICHD'] == 1][column].value_counts().sort_index()
                fig.add_trace(go.Bar(x=values_0.index, y=values_0.values, name=f'{column} (_MICHD 0)', marker_color='green'), row=current_row, col=current_col)
                fig.add_trace(go.Bar(x=values_1.index, y=values_1.values, name=f'{column} (_MICHD 1)', marker_color='red'), row=current_row, col=current_col)

        # Update the column and row for the next subplot
        current_col += 1
        if current_col > cols:
            current_col = 1
            current_row += 1

    # Adjust chart layout
    fig.update_layout(height=300 * rows, width=1200, title_text="Bivariate Analysis of Features with _MICHD", showlegend=False)

    # Show the figure in Streamlitt
    st.plotly_chart(fig, use_container_width=True)
