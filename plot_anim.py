import numpy as np
import pandas as pd
import numpy as np

import plotnine as p9
from plotnine.animation import PlotnineAnimation

# for animation in the notebook
import matplotlib as mpl
import matplotlib.pyplot as plt

# mpl.rc is the same as matplotlib.pyplot.rc - different namespaces.
# animation.html=html5 uses HTML5 video tag. Alternative is jshtml.
# For example the animation.writer is set to ffmpeg by default.
mpl.rc("animation", html="html5")

samples_count = 25 * 1

X_SPACE = (0, 2 * np.pi)

def plot(hot_sample):
    df = pd.DataFrame({
        "x": np.linspace(X_SPACE[0], X_SPACE[1], num=samples_count),
        "y": np.sin(np.linspace(X_SPACE[0], X_SPACE[1], num=samples_count)),
        "col": np.zeros(samples_count),
        "zeros": np.zeros(samples_count),
    })

    df.loc[hot_sample].at['col'] = 256 + 120

    def _breaks(limits):
        step_size = X_SPACE[1] / (samples_count - 1)
        breaks = np.arange(X_SPACE[0], X_SPACE[1] + step_size, step_size)
        return breaks

    def _labels(breaks):
        labels = []
        for i, b in enumerate(breaks):
            labels.append(str(i))
        return labels

    p = (
        p9.ggplot(df)
        + p9.geom_point(p9.aes("x", "y", color="col"), size=0.7)
        + p9.geom_point(p9.aes("x", "zeros", color="col"), size=0.1)
        + p9.scale_color_gradient(low="black", high="red")
        + p9.geom_vline(p9.aes(xintercept=[(hot_sample/(samples_count - 1)) * X_SPACE[1]]), alpha=0.2)
        + p9.scale_x_continuous(
            expand = (0, np.pi/8),
            # np.arange doesn't include the last value, thus add it to stop.
            breaks = _breaks,
            labels = _labels,
            minor_breaks = (lambda x: np.arange(x[0], x[1], np.pi / 8)),
        )
        + p9.theme_bw()
        + p9.theme(
            # plot_background=p9.element_rect(fill="#f0f0f0"),
            # figure_size is in inches
            figure_size=(12, 8)
        )
    )
    return p


# It is better to use a generator instead of a list
plots = (plot(k) for k in np.arange(0, samples_count))
ani = PlotnineAnimation(plots, interval=500, repeat_delay=500)
ani.save('./animation.mp4', dpi=80)
#ani
