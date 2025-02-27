import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter  # For GIF creation

import networkx as nx
import pandas as pd


def generate_well_connected_graph(ax, pros_cons):
    # Create a complete graph (every node is connected to every other node)
    pros, cons = pros_cons
    n = pros + cons + 1 # include utility
    G = nx.complete_graph(n)

    # Relabeling the nodes
    utility_node = n-1
    pro_nodes = [i+1 for i in range(pros)]
    con_nodes = [i+1 for i in range(pros, cons)]
    mapping = {i: (i + 1 if i != utility_node else "utility") for i in range(n)}
    G = nx.relabel_nodes(G, mapping)

    # Draw the graph on the given axis
    node_colors = ['lightgray' if node == "utility" 
                   else "pink" if node in pro_nodes 
                   else 'skyblue' for node in G.nodes()]
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=800, node_color=node_colors, font_size=9, font_color='black', font_weight='bold', edge_color='gray')
    return ax, pos, G

def animate_trans(j, outer_frame, e_axis, pos, market_outcomes):
    # Number of steps for each block to reach the target
    steps = 12

    for key, val in market_outcomes[outer_frame].items():
        seller, buyer = key
        quantity, price, total = val
        num_blocks = round(quantity)
        source_x, source_y = pos[seller][0], pos[buyer][1]
        target_x, target_y = pos[seller][0], pos[buyer][1]
        for block in range(num_blocks):
            if j >= block:
                # Draw energy blocks
                x = source_x + (target_x - source_x) * ((j-block) / steps)
                y = source_y + (target_y - source_y) * ((j-block) / steps)
                if (x != target_x) and (y != target_y) and (j-block <= steps):
                    e_axis.plot(x, y, 'go', markersize=7)

    # Redraw the annotation
    e_axis.text(0.5 * (pos[seller][0] + pos[buyer][0]),
                0.5 * (pos[seller][1] + pos[buyer][1]),
                f'{quantity} kWh @ ${price}/kWh\n${total}',
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=4)

def update_plot(frame, lines, variables, ax, outbox, orderbooks, market_outcomes):
    for (line_group, var_group) in zip(lines, variables):
        for line, var in zip(line_group, var_group):
            line.set_data(range(frame+1), var[:frame+1])
    ax[2].relim()
    ax[2].autoscale_view()
    ax[3].relim()
    ax[3].autoscale_view()

    def table_data_format(buyorders, sellorders):
        len_buy, len_sell = len(buyorders), len(sellorders)
        # transforming the shape of the data inserted to the data table
        if len_buy > len_sell:
            data = [[sellorders[i], buyorders[i]] for i in range(len_sell)]
            for i in range(len_sell, len_buy):
                data.append(["", buyorders[i]])
        elif len_sell > len_buy:
            data = [[sellorders[i], buyorders[i]] for i in range(len_sell)]
            for i in range(len_buy, len_sell):
                data.append([sellorders[i], ""])
        else:
            data = [[sellorders[i], buyorders[i]] for i in range(len_sell)]

        # modifying the row structure
        for i in range(len(data)):
            sell_order, buy_order = data[i]
            if sell_order and buy_order:
                sell_name, sell_price, sell_quant = sell_order
                buy_name, buy_price, buy_quant = buy_order
                data[i] = [f"agent{sell_name}: ${round(sell_price,2)}/kWh, {round(sell_quant,3)}kWh", f"agent{buy_name}: ${round(buy_price,2)}/kWh, {round(buy_quant,3)}kWh"]
            elif sell_order:
                sell_name, sell_price, sell_quant = sell_order
                data[i] = [f"agent{sell_name}: ${round(sell_price,2)}/kWh, {round(sell_quant,3)}kWh", ""]
            else:
                buy_name, buy_price, buy_quant = buy_order
                data[i] = ["", f"agent{buy_name}: ${round(buy_price,2)}/kWh, {round(buy_quant,3)}kWh"]

        return data
    
    market_frame = frame // 2
    buyorders, sellorders = orderbooks[market_frame]
    ax[0].set_title(f'Market Inputs / Outputs, hour {market_frame}')
    
    data = table_data_format(buyorders, sellorders)
    columns = ("Sell Orders", "Buy Orders")
    if not data:
        data = [["", ""],["", ""], ["", ""], ["", ""], ["", ""]]
    table = ax[0].table(cellText=data, colLabels=columns, colColours=("lavender", "lavender"), cellLoc='center', loc='center', bbox=[0, 0.4, 1, 0.5])
    # Adjust table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    outcome_text = ""
    for key, val in market_outcomes[market_frame].items():
        seller, buyer = key
        quantity, price, total = val
        if seller != "utility":
            if buyer != "utility":
                outcome_text += f"{seller} --> {buyer} = {quantity}kWh @{price}/kWh, ${total} total\n"
            else:
                outcome_text += f"{seller} --> utility = {quantity}kWh @{price}/kWh, ${total} total\n"
        else:
            outcome_text += f"utility --> agent {buyer} = {quantity}kWh @{price}/kWh, ${total} total\n"

    outbox.set_text(outcome_text, )

    return [line for linegroup in lines for line in linegroup] + [table] + [outbox]

def figure_setup(pros_cons, money_records, energy_records):
    time_steps = 48
    pros, cons = pros_cons
    n = pros + cons
    # Initialize variables
    variables = [
        money_records,
        energy_records,      # energy records
    ]
    # Create figure and axes
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    
    ax = ax.flatten()  # Flatten the array to easily iterate over subplots

    # Create lines for each plotting subplot
    lines = [[ax[i+2].plot([], [], label=f'agent {_}')[0] for _ in range(n)] for i in range(2)]

    # Set up the text display subplot
    ax[0].tick_params(axis='both', which='both',
                    bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)
    # order_box = ax[0].text(0.5, 0.8, '', horizontalalignment='center', verticalalignment='center')
    outbox = ax[0].text(0.5, 0.15, '', horizontalalignment='center', verticalalignment='center', fontsize='medium')
    ax[0].set_title('Market Inputs / Outputs')
    # Create the table
    table = ax[0].table(cellText=[["", ""],["", ""], ["", ""], ["", ""], ["", ""]], colLabels=("Sell Orders", "Buy Orders"), colColours=['lavender', 'lavender'],cellLoc='center', loc='center', bbox=[0, 0.4, 1, 0.5]) 
    # Adjust table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    # Add a title above the table
    ax[0].text(0.5, 0.95, 'Orderbooks', horizontalalignment='center', verticalalignment='center', fontsize=10, fontweight='bold')
    ax[1], pos, G = generate_well_connected_graph(ax[1], pros_cons)

    ax[1].set_title('Energy Transfer')

    def setup_plots(ax2, ax3):
    # Set labels for each plotting subplot
        def setup_dotted(axis):
            axis.axhline(y=0, color='black', linestyle=':')
            for i in range(48):
                if (i%2 == 0) and (i != 0):
                    axis.axvline(x=i, color='green', linestyle=':')
        ax2.set_xlim(0, time_steps)
        ax2.set_xlabel('Time (half hour increments)')
        ax2.set_title('Money')
        ax2.legend(loc='lower left')
        setup_dotted(ax2)

        ax3.set_xlim(0, time_steps)
        ax3.set_xlabel('Time (half hour increments)')
        ax3.set_title('Energy Balance')
        # ax3.legend(loc='upper right')
        setup_dotted(ax3)

    # Setup Money and Energy Balance plots
    setup_plots(ax[2], ax[3])

    return (fig, pos, ax, variables, lines, table, outbox)

# fig, pos, ax, variables, lines, table, outbox = figure_setup(pros_cons, money_records, energy_records)
# # energytrans_vars = (market_outcomes, ax[1], pos, block_lines)
# ani = animation.FuncAnimation(
#     fig, update_plot, fargs=(lines, variables, ax[0], outbox, orderbooks, market_outcomes), 
#     frames=48, interval=2000, blit=True
# )

# # save as a gif
# # ani.save('my_animation.gif', writer=PillowWriter(fps=30))

# # # save as a video
# # Writer = animation.writers['ffmpeg']
# # writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
# # ani.save('new_simulation.mp4', writer=writer)

# plt.tight_layout()
# plt.show()




# ATTEMPTED TO CREATE AN ANIMATION
# # Function to update the plot
# def update_plot(frame, temperature, pressure, humidity, line_temp, line_press, line_humid):
#     # Simulate data
#     new_temp = temperature[-1] + np.random.randn() * 0.5
#     new_press = pressure[-1] + np.random.randn() * 0.1
#     new_humid = humidity[-1] + np.random.randn() * 0.2

#     # Append new data
#     temperature.append(new_temp)
#     pressure.append(new_press)
#     humidity.append(new_humid)

#     # Update lines
#     line_temp.set_data(range(len(temperature)), temperature)
#     line_press.set_data(range(len(pressure)), pressure)
#     line_humid.set_data(range(len(humidity)), humidity)

#     # Update axes limits
#     ax.relim()
#     ax.autoscale_view()

#     return line_temp, line_press, line_humid

# # Initialize variables
# time_steps = 100
# temperature = [20]  # Starting temperature in °C
# pressure = [1013]   # Starting pressure in hPa
# humidity = [50]     # Starting humidity in %

# # Create figure and axes
# fig, ax = plt.subplots()
# line_temp, = ax.plot([], [], label='Temperature (°C)')
# line_press, = ax.plot([], [], label='Pressure (hPa)')
# line_humid, = ax.plot([], [], label='Humidity (%)')

# # Set plot limits and labels
# ax.set_xlim(0, time_steps)
# ax.set_ylim(0, 100)
# ax.set_xlabel('Time')
# ax.set_ylabel('Value')
# ax.legend()

# # Animate
# ani = animation.FuncAnimation(
#     fig, update_plot, fargs=(temperature, pressure, humidity, line_temp, line_press, line_humid), 
#     frames=time_steps, interval=200, blit=True
# )

# save as a gif
# ani.save('my_animation.gif', writer=PillowWriter(fps=30))

# save as a video
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
# ani.save('simulation.mp4', writer=writer)

# plt.show()


# def update_plot(frame, lines, variables, textboxes, energy_trans_vars):
#     """
#     energy_trans_vars = (market_outcomes, energynet_ax, pos)
#     Variables = [
#                     [money_records],
#                     [energy_records]
#                 ]

#         Lines = [
#                     [money_lines], # line group
#                     [energy_lines]
#                 ]
#     Updates the whole figure of 4 subplots every 30 minutes
#         1. Money lines of all agents [money1, money2, ...]
#             money1 = new money_
#         2. Energy lines of all agents [energy1, energy2, ...]
#         3. Energy transfer animation
#         4. Market period orderbooks + outcome simulation

#     Args:
#         frame: variable to move frames used in animation function
#         money_lines: [list holding all money lines of each agent]
#         energy_lines: [list holding all energy lines of each agent]
#         money_records: [[money data for agent1], [], ...] -- money recordings for each agent
#         energy_records: [[energy data for agent1], [], ...]
#         orderbook:
#         outcome:
#     """

#     # update the lines for the first 2 subplots
#     for i, (line_group, var_group) in enumerate(zip(lines, variables)):
#         for line, var in zip(line_group, var_group):
#             line.set_data(range(frame+1), var[:frame+1])
#         # ax[i+2].relim()
#         # ax[i+2].autoscale_view()
#     global fig
#     global pos
#     # Update Energy Transfer figure
#     market_outcomes, energynet_ax, pos = energy_trans_vars
#     # Function to animate the transfer

#     # Create animation
#     if frame%2 == 0:
#         outer_frame = frame//2
#         animate_trans(frame)
#         energy_ani = animation.FuncAnimation(fig, animate_trans, fargs=(outer_frame, energynet_ax, pos, market_outcomes), frames=12, interval=200, repeat=False)
#     # Update text
#     order_box, out_box = textboxes
#     orderbook_text = f"Orderbook\n{frame}"
#     order_box.set_text(orderbook_text)
#     outcome_text = f"Market Clearing Outcome\n{frame}"
#     out_box.set_text(outcome_text)
#     # Update text
#     # new_text = f"Frame: {frame}"
#     # textbox.set_text(new_text)

#     return [line for linegroup in lines for line in linegroup] + [order_box, out_box]