import matplotlib.pyplot as plt


def plot_qc_map(quebec):
    if quebec.empty:
        raise ValueError("no map loaded.")
    coo = []
    i = -57

    while i > -80.0:
        i -= 0.25
        j = 64.0
        while j > 45.0:
            j -= 0.25
            coo.append((i, j))
    fig, ax = plt.subplots(figsize=(24, 28))
    quebec.boundary.plot(ax=ax, color='black', linewidth=1)
    latitude_lines = [lat * 0.25 - 0.125 for lat in range(45 * 4, 64 * 4)]
    longitude_lines = [lon * 0.25 - 0.125 for lon in range(-80 * 4, -57 * 4)]

    for lat in latitude_lines:
        ax.plot([lon for lon, lat_ in coo], [lat] * len([lon for lon, lat_ in coo]), color='gray', linestyle='-')

    for lon in longitude_lines:
        ax.plot([lon] * len([lat for lon, lat in coo]), [lat for lon, lat in coo], color='gray', linestyle='-')


    label_lon = [x for x in range(-80, -56, 1)]
    label_lat = [x for x in range(45, 65, 1)]
    ax.set_xticks(label_lon)
    ax.set_yticks(label_lat)
    ax.set_xticklabels(label_lon, fontsize=14)
    ax.set_yticklabels(label_lat, fontsize=14)

    ax.set_xlabel("Longitude", fontsize=16)
    ax.set_ylabel("Latitude", fontsize=16)

    ax.set_aspect('equal')


    ax.set_xlim([-80, -56])
    ax.set_ylim([45, 64])
    return(fig,ax)