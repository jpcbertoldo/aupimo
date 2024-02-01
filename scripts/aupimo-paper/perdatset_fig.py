"""TODO REFACTOR"""

# %%
# TEX FILE

TEX_TEMPLATE = r"""
\begin{figure}[ht]
    \centering
    \begin{subfigure}[b]{\linewidth}
      \includegraphics[width=\linewidth,valign=t,keepaspectratio]{src/img/perdataset/perdataset_DATASETIDX_table.pdf}
      \caption{Statistics and pairwise statistical tests.}
      \label{fig:benchmark-DATASETIDX-table}
    \end{subfigure}
    \\ \vspace{2mm}
    \begin{subfigure}[b]{0.5\linewidth}
      \includegraphics[width=\linewidth,valign=t,keepaspectratio]{src/img/perdataset/perdataset_DATASETIDX_diagram.pdf}
      \caption{Average rank diagram.}
      \label{fig:benchmark-DATASETIDX-diagram}
    \end{subfigure}
    \\ \vspace{2mm}
    \begin{subfigure}[b]{0.49\linewidth}
      \includegraphics[width=\linewidth,valign=t,keepaspectratio]{src/img/perdataset/perdataset_DATASETIDX_boxplot.pdf}
      \caption{Score distributions.}
      \label{fig:benchmark-DATASETIDX-boxplot}
    \end{subfigure}
    ~
    \begin{subfigure}[b]{0.49\linewidth}
      \includegraphics[width=\linewidth,valign=t,keepaspectratio]{src/img/perdataset/perdataset_DATASETIDX_curves.pdf}
      \caption{PIMO curves.}
      \label{fig:benchmark-DATASETIDX-pimo-curves}
    \end{subfigure}
    \\  \vspace{2mm}
    \begin{subfigure}[b]{\linewidth}

      \begin{minipage}{\linewidth}
        \centering
        \includegraphics[height=32mm,valign=t,keepaspectratio]{src/img/heatmaps/DATASETIDX/000.jpg}
        \includegraphics[height=32mm,valign=t,keepaspectratio]{src/img/heatmaps/DATASETIDX/001.jpg}
        \includegraphics[height=32mm,valign=t,keepaspectratio]{src/img/heatmaps/DATASETIDX/002.jpg}
      \end{minipage}
      \\
      \begin{minipage}{\linewidth}
        \centering
        \includegraphics[height=32mm,valign=t,keepaspectratio]{src/img/heatmaps/DATASETIDX/003.jpg}
        \includegraphics[height=32mm,valign=t,keepaspectratio]{src/img/heatmaps/DATASETIDX/004.jpg}
        \includegraphics[height=32mm,valign=t,keepaspectratio]{src/img/heatmaps/DATASETIDX/005.jpg}
      \end{minipage}
      \caption{
        Heatmaps.
        Images selected according to AUPIMO's statistics.
        Statistic and image index annotated on upper left corner.
      }
      \label{fig:benchmark-DATASETIDX-heatmap}
    \end{subfigure}
    \caption{
      Benchmark on DATASETLABEL.
      PIMO curves and heatmaps are from MODELLABEL.
      NUMIMAGES images (NUMNORMALIMAGES normal, NUMANOMALYIMAGES anomalous).
    }
    \label{fig:benchmark-DATASETIDX}
\end{figure}
"""

# `dc` stands for dataset/category
for dcidx, dcrow in dc_bestmodel_args.iterrows():
    # args
    dataset, category = dcrow[["dataset", "category"]]
    best_model = dcrow["best_model"]
    print(f"{dcidx=} {dataset=} {category=} {best_model=}")
    # =======================================================================
    # data
    modeldf = data.loc[dataset, category, best_model]
    curves = PImOResult.load(modeldf["aupimo_curves_fpath"])
    imgclass = curves.image_classes
    num_images = len(imgclass)
    num_normal_images = (imgclass == 0).sum().item()
    num_anomaly_images = (imgclass == 1).sum().item()
    # break
    # =======================================================================
    # write tex file
    tex = TEX_TEMPLATE.replace("DATASETIDX", f"{dcidx:03}")
    tex = tex.replace("DATASETLABEL", f"{DATASETS_LABELS[dataset]} / {CATEGORIES_LABELS[category]}")
    tex = tex.replace("MODELLABEL", constants.MODELS_LABELS[best_model])
    tex = tex.replace("NUMIMAGES", f"{num_images:03}")
    tex = tex.replace("NUMNORMALIMAGES", f"{num_normal_images:03}")
    tex = tex.replace("NUMANOMALYIMAGES", f"{num_anomaly_images:03}")
    (PERDATASET_FIGS_SAVEDIR / f"{dcidx:03}.tex").write_text(tex)
    break
