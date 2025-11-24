import time
import urllib.parse
import numpy as np
import pandas as pd
import plotly.express as px
import umap  
from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget  
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------------------
try:
    df = pd.read_parquet('methylation_data.parquet')
    meta = pd.read_parquet('metadata.parquet')
    
    # Pre-calculate globals to save time later
    samples = list(df.index.astype(str))
    
    # Global variance (calculated once on startup)
    global_var = df.var(axis=0).sort_values(ascending=False)
    global_var_order = global_var.index.tolist()
    
    n_features = df.shape[1]
    n_samples_total = df.shape[0]
    
    # Get unique diagnoses for the dropdown
    uniq_dx = sorted(meta['Diagnosis'].fillna("NA").astype(str).unique())
    sample_groups = ['All'] + uniq_dx

except FileNotFoundError:
    print("CRITICAL: Data files not found. Please run process_data.py first.")
    # Fallbacks to prevent import crash
    df = pd.DataFrame()
    meta = pd.DataFrame()
    samples, global_var_order, sample_groups = [], [], []
    n_features, n_samples_total = 100, 10

# -------------------------------------------------------------------------
# 2. Logic Functions
# -------------------------------------------------------------------------
def compute_umap(df, selected_ids, top_n, n_neighbors, min_dist, metric, 
                 pca_n, variance_mode, global_var_order_list, normalization, random_state):
    """
    Core computational logic. Separated from server for cleanliness.
    """
    # 1. Subset Data
    Xdf = df.reindex(selected_ids)
    
    # 2. Normalization
    if normalization == "Beta values":
        # Center only
        scaler = StandardScaler(with_mean=True, with_std=False)
        X_vals = scaler.fit_transform(Xdf)
    elif normalization == "M-values":
        # Clip to [0.000001, 0.999999]
        X_safe = Xdf.values.clip(1e-6, 1.0 - 1e-6)
        X_vals = np.log2(X_safe / (1.0 - X_safe))
    else:
        X_vals = Xdf.values

    X_normalized = pd.DataFrame(X_vals, index=Xdf.index, columns=Xdf.columns)

    # 3. Feature Selection
    if variance_mode == "Global variance":
        top_features = global_var_order_list[:int(top_n)]
    else:
        # Calculate variance on the subset
        var_order = X_normalized.var(axis=0).sort_values(ascending=False).index
        top_features = list(var_order[:int(top_n)])

    Xsel = X_normalized.loc[:, top_features]

    # 4. PCA Pre-processing (Optional but recommended)
    max_possible_pcs = min(Xsel.shape[0], Xsel.shape[1])
    n_pcs = min(int(pca_n), max_possible_pcs)
    
    if n_pcs >= 2 and Xsel.shape[1] > n_pcs:
        pca = PCA(n_components=n_pcs, random_state=42)
        Xp = pca.fit_transform(Xsel)
    else:
        Xp = Xsel

    # 5. UMAP
    eff_n_neighbors = min(int(n_neighbors), Xp.shape[0] - 1)
    if eff_n_neighbors < 2: eff_n_neighbors = 2
    
    reducer = umap.UMAP(
        n_neighbors=eff_n_neighbors,
        n_components=2,
        min_dist=float(min_dist),
        metric=metric,
        random_state=int(random_state),
        low_memory=True,
        n_jobs=1
    )
    
    emb = reducer.fit_transform(Xp)
    return emb, selected_ids

def create_legend_html(meta):
    """Generates the static HTML legend."""
    if meta.empty: return ""
    html = ['<div style="font-size:13px; padding:10px; border:1px solid #eee; border-radius:5px; background:#f9f9f9;">']
    
    for col, title in [('Color_dx', 'Diagnosis'), ('Color_WHO', 'WHO Category')]:
        if col in meta.columns:
            html.append(f'<div style="margin-bottom:8px;"><strong>{title}</strong></div>')
            html.append('<div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:15px;">')
            
            # Get unique label-color pairs
            # We assume the column "Dx" corresponds to "Color_dx" and "WHO..." to "Color_WHO"
            label_col = 'Dx' if col == 'Color_dx' else 'WHO_differentiation'
            pairs = meta[[label_col, col]].drop_duplicates().sort_values(label_col)
            
            for _, row in pairs.iterrows():
                color = row[col]
                label = row[label_col]
                html.append(
                    f'<div style="display:flex; align-items:center;">'
                    f'<span style="width:12px;height:12px;background:{color};margin-right:4px;border:1px solid #ccc;"></span>'
                    f'<span>{label}</span></div>'
                )
            html.append('</div>')
            
    html.append('</div>')
    return "".join(html)

# -------------------------------------------------------------------------
# 3. UI
# -------------------------------------------------------------------------
UMAP_CACHE = {}
MAX_CACHE = 20
sidebar_width = 380

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Configuration"),
        ui.input_radio_buttons("active_plot", "Target Plot:", ["Plot A (Left)", "Plot B (Right)"]),
        ui.hr(),
        ui.input_select("preset", "Presets", ["Custom", "Fast Preview", "High Detail"], selected="Custom"),
        ui.input_radio_buttons("normalization", "Normalization", ["M-values", "Beta values"]),
        
        ui.hr(),
        ui.tags.label("Features (CpGs)"),
        ui.layout_column_wrap(
             ui.input_slider("top_n", None, min=100, max=n_features, value=1000, step=100, width="100%"),
             width=1
        ),
        ui.tags.label("Neighbors"),
        ui.input_slider("n_neighbors", None, min=2, max=100, value=15, step=1),
        
        ui.tags.label("Min Dist"),
        ui.input_slider("min_dist", None, 0, 1, 0.1, step=0.05),
        
        ui.accordion(
            ui.accordion_panel("Advanced Settings",
                ui.input_slider("n_components", "PCA Components", 5, 100, 50, step=5),
                ui.input_select("metric", "Distance Metric", ["euclidean", "cosine", "correlation"]),
                ui.input_radio_buttons("variance_mode", "Variance Calculation", ["Subset variance", "Global variance"]),
                ui.input_action_button("randomize", "🎲 Random Seed"),
                ui.output_text("current_seed_display")
            ),
            open=False
        ),
        
        ui.hr(),
        ui.input_selectize("sample_groups", "Filter Samples", choices=sample_groups, multiple=True, selected=["All"]),
        ui.output_text("sample_count_text"),
        
        ui.hr(),
        ui.input_radio_buttons("color_col", "Color Mapping", ["Diagnosis", "WHO Category"]),
        
        ui.hr(),
        ui.input_action_button("run", "RUN UMAP", class_="btn-primary btn-lg", width="100%"),
        ui.br(),
        ui.output_text("status_text"),
        ui.output_text("compute_time"),
        
        ui.hr(),
        ui.h5("Export Data"),
        ui.output_ui("download_links"),
        
        width=sidebar_width
    ),
    
    # Main Layout
    ui.row(
        # We use output_widget (shinywidgets) for much better performance than standard HTML
        ui.column(6, ui.card(ui.card_header("Plot A"), output_widget("plot_a_widget"))),
        ui.column(6, ui.card(ui.card_header("Plot B"), output_widget("plot_b_widget")))
    ),
    ui.br(),
    ui.card(
        ui.card_header("Legend"),
        ui.div(ui.HTML(create_legend_html(meta)))
    ),
    title="Sarcoma Methylation Atlas"
)

# -------------------------------------------------------------------------
# 4. Server
# -------------------------------------------------------------------------
def server(input, output, session):
    # State
    # Store the *Figure* object for the widgets
    plot_a_fig = reactive.Value(None)
    plot_b_fig = reactive.Value(None)
    
    data_a = reactive.Value(None)
    data_b = reactive.Value(None)
    
    status = reactive.Value("Ready")
    comp_time = reactive.Value("")
    
    # Internal reactive value for random state
    current_seed = reactive.Value(42)

    @reactive.Effect
    @reactive.event(input.randomize)
    def _randomize():
        current_seed.set(np.random.randint(0, 10000))
        
    @render.text
    def current_seed_display():
        return f"Seed: {current_seed.get()}"

    @reactive.Effect
    @reactive.event(input.preset)
    def _apply_preset():
        p = input.preset()
        if p == "Fast Preview":
            ui.update_slider("top_n", value=500)
            ui.update_slider("n_neighbors", value=15)
            ui.update_slider("min_dist", value=0.5)
        elif p == "High Detail":
            ui.update_slider("top_n", value=5000)
            ui.update_slider("n_neighbors", value=30)
            ui.update_slider("min_dist", value=0.1)

    @reactive.Calc
    def get_selected_ids():
        sel = input.sample_groups()
        if not sel or "All" in sel:
            return samples
        # Map diagnosis strings to IDs
        return meta[meta['Diagnosis'].isin(sel)]['ID'].tolist()

    @render.text
    def sample_count_text():
        n = len(get_selected_ids())
        return f"{n} samples selected"

    @reactive.Effect
    @reactive.event(input.run)
    def _compute():
        start = time.time()
        status.set("⏳ Computing... (this may take a moment)")
        comp_time.set("")
        
        try:
            sel_ids = get_selected_ids()
            if len(sel_ids) < 3:
                raise ValueError("Please select at least 3 samples.")
                
            # Create Cache Key
            params = (
                tuple(sorted(sel_ids)), input.top_n(), input.n_neighbors(),
                input.min_dist(), input.metric(), input.n_components(),
                input.variance_mode(), input.normalization(), current_seed.get()
            )
            
            if params in UMAP_CACHE:
                emb, ids = UMAP_CACHE[params]
                status.set("✅ Loaded from cache")
            else:
                emb, ids = compute_umap(
                    df, sel_ids, input.top_n(), input.n_neighbors(),
                    input.min_dist(), input.metric(), input.n_components(),
                    input.variance_mode(), global_var_order, input.normalization(),
                    current_seed.get()
                )
                
                # Update Cache
                if len(UMAP_CACHE) > MAX_CACHE:
                    del UMAP_CACHE[next(iter(UMAP_CACHE))]
                UMAP_CACHE[params] = (emb, ids)
                status.set("✅ Computation Complete")

            # Merge with metadata for plotting
            plot_df = pd.DataFrame(emb, index=ids, columns=['UMAP1', 'UMAP2'])
            plot_df = plot_df.join(meta.set_index('ID'))
            plot_df.reset_index(names='ID', inplace=True)
            
            # Determine Color Column
            c_col = 'Color_dx' if input.color_col() == "Diagnosis" else 'Color_WHO'
            
            # Build Plotly Figure
            fig = px.scatter(
                plot_df, x="UMAP1", y="UMAP2",
                color=c_col, color_discrete_map="identity",
                hover_data=["ID", "Diagnosis", "WHO_differentiation"],
                width=None, height=600 # Let container determine width
            )
            fig.update_traces(marker=dict(size=6, opacity=0.8))
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                margin=dict(l=0,r=0,t=30,b=0),
                showlegend=False
            )
            
            # Update target
            if input.active_plot() == "Plot A (Left)":
                plot_a_fig.set(fig)
                data_a.set(plot_df)
            else:
                plot_b_fig.set(fig)
                data_b.set(plot_df)
                
            elapsed = time.time() - start
            comp_time.set(f"Time: {elapsed:.2f}s")
            
        except Exception as e:
            status.set(f"Error: {str(e)}")

    # Render Widgets (The optimized way)
    @render_widget
    def plot_a_widget():
        return plot_a_fig.get() if plot_a_fig.get() else px.scatter(title="Waiting for run...")

    @render_widget
    def plot_b_widget():
        return plot_b_fig.get() if plot_b_fig.get() else px.scatter(title="Waiting for run...")

    @render.ui
    def download_links():
        links = []
        for name, data in [("Plot A", data_a.get()), ("Plot B", data_b.get())]:
            if data is not None:
                csv = data.to_csv(index=False)
                b64 = urllib.parse.quote(csv)
                href = f"data:text/csv;charset=utf-8,{b64}"
                links.append(
                    f'<a class="btn btn-sm btn-outline-secondary" '
                    f'href="{href}" download="{name.replace(" ", "_")}.csv" '
                    f'style="margin-right:10px;">⬇ {name} Data</a>'
                )
        return ui.HTML("".join(links)) if links else "No data generated yet."

app = App(app_ui, server)