import numpy as np
import pandas as pd
import numpy as np

import plotnine as p9
from plotnine.animation import PlotnineAnimation

# for animation in the notebook
from matplotlib import rc

#rc("animation", html="html5")


samples_count = 24

def plot(hot_sample):
    df = pd.DataFrame({
        "x": np.linspace(0, 2 * np.pi, num=samples_count),
        "y": np.sin(np.linspace(0, 2 * np.pi, num=samples_count)),
        "col": np.zeros(samples_count),
        "zeros": np.zeros(samples_count),
    })

    df["col"][hot_sample] = 120

    p = (
        p9.ggplot(df)
        + p9.geom_point(p9.aes("x", "y", color="col"), size=0.7)
        + p9.geom_point(p9.aes("x", "zeros", color="col"), size=0.1)
        + p9.scale_color_gradient(low="black", high="red")
        + p9.geom_vline(p9.aes(xintercept=(hot_sample/samples_count) * 2 * np.pi, alpha=0.2))
        + p9.lims(
            # All the plots have scales with the same limits
            x=(0, 2 * np.pi),
            y=(-1, 1)
        )
        # + p9.theme_matplotlib()
        + p9.theme(
            # plot_background=p9.element_rect(fill="#f0f0f0"),
            figure_size=(12, 8)
        )
    )
    return p


# It is better to use a generator instead of a list
plots = (plot(k) for k in np.arange(0, samples_count))
ani = PlotnineAnimation(plots, interval=250, repeat_delay=500)
ani.save('./animation.mp4', dpi=150)
#ani
