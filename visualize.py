import string
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
import numpy as np

def visualize(warehouses_location, warehouses_initial_capacity, all_warehouses, all_customers, all_capacities):

    warehouse_colors = ['#2b83ba','#fdae61','#abdda4','#d7191c','#ffffbf']
    
    # Create a figure with two subplots: one for the map and one for the bar plot
    fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(12, 6))

    # Create a title for the figure
    #fig.suptitle("Warehouse Inventory Management Simulation", fontsize=16)
    
    # Set up the map (left side)
    ax_map.set_aspect('equal', adjustable='box')
    ax_map.set_xlim(-100, 100)
    ax_map.set_ylim(-100, 100)
    ax_map.set_xticks(np.arange(-100, 101, 50))
    ax_map.set_yticks(np.arange(-100, 101, 50))
    ax_map.grid(True)
    
    # Set up the bar plot (right side)
    ax_bar.set_xlim(0, len(warehouses_location)+1)
    ax_bar.set_ylim(0, max(warehouses_initial_capacity) * 1.2)  # Allow some space above the max capacity
    ax_bar.set_ylabel("Facility capacity")
    ax_bar.set_xticks(np.arange(1,len(warehouses_location)+1))
    ax_bar.set_xticklabels([f"F {i}" for i in range(1, len(warehouses_location)+1)])
    
    # Initialize the map with warehouses and customers
    def init():
        # Add warehouses to the map
        for j in range(len(warehouses_location)):
            ax_map.add_patch(patches.Circle(warehouses_location[j], 5, fill=True, color=warehouse_colors[j]))
        
        # Initialize the bar plot
        bars = ax_bar.bar(np.arange(1,len(warehouses_initial_capacity)+1), warehouses_initial_capacity, color=warehouse_colors[:len(warehouses_location)])
        
        return []

    # Animate function to update both the map and the bar plot
    def animate(i):
        # Clear the previous customer and update warehouse positions on the map
        #for patch in ax_map.patches:
        #    patch.remove()
        
        # Add warehouses again (do not remove them every frame)
        for j in range(len(warehouses_location)):
            ax_map.add_patch(patches.Circle(warehouses_location[j], 5, fill=True, color=warehouse_colors[j]))
        
        # Add current customer
        ax_map.add_patch(patches.Circle(all_customers[i][1:3], 2, fill=False, color=warehouse_colors[all_warehouses[i]]))
        
        # Update the bar plot with current capacities
        current_capacities = [all_capacities[i][j] for j in range(len(warehouses_location))]
        
        # Clear previous bar plot and redraw it with updated values
        ax_bar.clear()
        ax_bar.set_xlim(0, len(warehouses_location)+1)
        ax_bar.set_ylim(0, max(warehouses_initial_capacity) * 1.2)  # Allow some space above the max capacity
        ax_bar.set_ylabel("Facility capacity")
        ax_bar.set_xticks(np.arange(1,len(warehouses_location)+1))
        ax_bar.set_xticklabels([f"F {i}" for i in range(1,len(warehouses_location)+1)])
        ax_bar.bar(np.arange(1,len(warehouses_initial_capacity)+1), current_capacities, color=warehouse_colors[:len(warehouses_location)])

        return []

    # Create the animation (disable blit=True here to avoid errors)
    ani = animation.FuncAnimation(fig, animate, frames=len(all_customers), interval=500, init_func=init)

    #Store the animation as a gif
    ani.save('visualization_examples/warehouse_simulation.gif', writer='pillow', fps=2)

    plt.show()


def visualize_assignments(warehouses_location, all_warehouses, all_customers):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set up the axis limits and aspect
    ax.set_aspect('equal')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    
    warehouse_colors = ['#2b83ba','#fdae61','#abdda4','#d7191c','#ffffbf']

    # Plot the warehouses
    for j in range(len(warehouses_location)):
        ax.add_patch(patches.Circle(warehouses_location[j], 7, fill=True, color=warehouse_colors[j]))
        #Print facility A or B
        ax.text(warehouses_location[j][0], warehouses_location[j][1] + 9, f"Facility {chr(65+j)}", ha='center', fontsize=12, color='black')
    
    # Plot the customers and assign them to the corresponding warehouse
    for i in range(len(all_customers)):
        customer_location = all_customers[i][1:3]
        assigned_warehouse = all_warehouses[i]
        ax.plot(customer_location[0], customer_location[1], marker='x', color=warehouse_colors[assigned_warehouse], markersize=12, markeredgewidth=4)
        #ax.text(customer_location[0] + 2, customer_location[1], f'C{i}', fontsize=8, color=warehouse_colors[assigned_warehouse])

    # Save the plot as an image file
    plt.grid(False)
    plt.savefig('visualization_examples/warehouse_assignments.png')  # Save the plot to a file
    plt.show()  # Display the plot

def generate_warehouse_visualization():
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))  # 2 rows, 4 columns

    warehouse_colors = ['#2b83ba','#fdae61','#abdda4','#d7191c','#ffffbf']
    num_warehouses_list = [2, 3, 4, 5]

    for row, distribution_type in enumerate(["uniform", "uneven"]):
        for col, num_warehouses in enumerate(num_warehouses_list):
            ax = axes[row, col]
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            ax.set_aspect('equal', adjustable='box')
            if row == 0:
                ax.set_title(f"{num_warehouses} facilities", fontsize=12, fontweight='bold')
            ax.set_xticks(np.arange(-100, 101, 50))
            ax.set_yticks(np.arange(-100, 101, 50))
            ax.grid(True, linestyle='--', linewidth=0.5)

            # Generate warehouse locations
            np.random.seed(42 + row + col)  # Ensure reproducibility
            warehouses_location = np.random.uniform(-80, 80, (num_warehouses, 2))

            if num_warehouses == 2:
                warehouses_location = [[-50, -50], [50, 50]]
                if distribution_type == "uniform":
                    capacities = [50, 50]
                else:
                    capacities = [70, 30]
            elif num_warehouses == 3:
                warehouses_location = [[-50, -50], [0, 50], [50, -50]]
                if distribution_type == "uniform":
                    capacities = [33, 33, 33]
                else:
                    capacities = [50, 30, 20]
            elif num_warehouses == 4:
                warehouses_location = [[-50, -50], [-50, 50], [50, 50], [50, -50]]
                if distribution_type == "uniform":
                    capacities = [25, 25, 25, 25]
                else:
                    capacities = [40, 30, 20, 10]
            elif num_warehouses == 5:
                warehouses_location = [[-50, -50], [-50, 50], [50, 50], [50, -50], [0, 0]]
                if distribution_type == "uniform":
                    capacities = [20, 20, 20, 20, 20]
                else:
                    capacities = [30, 25, 20, 15, 10]
            # Plot warehouses
            facility_letters = string.ascii_uppercase
            for i, (loc, cap) in enumerate(zip(warehouses_location, capacities)):
                ax.add_patch(patches.Circle(loc, 11, fill=True, facecolor=warehouse_colors[i % len(warehouse_colors)], edgecolor='black', linewidth=1.0, zorder=2))
                label = ax.text(loc[0], loc[1], facility_letters[i % len(facility_letters)], ha='center', va='center', fontsize=11, color='white', zorder=3, fontweight='bold')
                label.set_path_effects([path_effects.withStroke(linewidth=0.5, foreground='black')])
                ax.text(loc[0], loc[1] + 15, f"{cap}%", ha='center', fontsize=12, color='black')

        # Add distribution type text on the left side
        ax = axes[row, 0]
        ax.text(-132, 0, f"{distribution_type.capitalize()} distribution", fontsize=12, ha='center', va='center', rotation=90, fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualization_examples/facilities.pdf', dpi=600)
    plt.show()


def generate_myopic_vs_lookahead_visualization():
    from env import InventoryEnv
    from policies.myopic_policy import MyopicPolicy
    from policies.exact_value_function_policy import ExactValueFunctionPolicy

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    warehouse_colors = ['#2b83ba', '#fdae61', '#abdda4', '#d7191c', '#ffffbf']
    capacity_distribution_settings = [[25, 25], [24, 6], [6, 24]]
    warehouses_location = [[-50, -50], [50, 50]]

    # act() only reads env.grid_size, so one shared env instance is enough for every
    # cell/policy below regardless of that cell's own capacity distribution
    env = InventoryEnv(num_warehouses=2, num_customers=50,
                       capacity_distribution='uniform', grid_size=200)
    policies = {
        "myopic": MyopicPolicy(env),
        "optimal": ExactValueFunctionPolicy(env, num_warehouses=2, num_customers=50, capacity_distribution='uniform')
    }

    def make_state(position, warehouses_capacity):
        return {
            'static_info': {'num_warehouses': 2, 'warehouses_location': warehouses_location},
            'warehouses_capacity': list(warehouses_capacity),
            'new_customer': [0, position[0], position[1], 1],
            'customers_left': sum(warehouses_capacity),
        }

    for row, distribution_type in enumerate(["myopic", "optimal"]):
        policy = policies[distribution_type]

        for col, capacity_distribution in enumerate(capacity_distribution_settings):
            ax = axes[row, col]
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks(np.arange(-100, 101, 50))
            ax.set_yticks(np.arange(-100, 101, 50))

            square = 2
            for i in range(-100 + int(square / 2), 101 - int(square / 2), int(square)):
                for j in range(-100 + int(square / 2), 101 - int(square / 2), int(square)):
                    state = make_state((i, j), capacity_distribution)
                    action = policy.act(state)
                    color = warehouse_colors[action]
                    ax.add_patch(plt.Rectangle((i - square / 2, j - square / 2), square, square,
                                               fill=True, color=color, alpha=0.3))

            for i, loc in enumerate(warehouses_location):
                ax.add_patch(patches.Circle(loc, 12, fill=True,
                                            facecolor=warehouse_colors[i % len(warehouse_colors)],
                                            edgecolor='black', linewidth=1.0, zorder=2))
                warehouse_name = f"{chr(65 + i)}"
                label = ax.text(loc[0], loc[1] - 3, warehouse_name, ha='center', fontsize=16,
                                color='white', fontweight='bold', zorder=3)
                label.set_path_effects([path_effects.withStroke(linewidth=0.5, foreground='black')])

            # Row label: policy name, attached to the left-hand panel
            if col == 0:
                ax.set_ylabel(f"{distribution_type.capitalize()}\npolicy",
                              fontsize=15, labelpad=8)

            # Column header: capacity distribution, attached above the top row
            if row == 0:
                ax.text(0.5, 1.16, "Capacity distribution", fontsize=15,
                        ha='center', va='bottom', transform=ax.transAxes)
                ax.text(0.3, 1.05, f"{capacity_distribution[0]/(capacity_distribution[0] + capacity_distribution[1]) * 100:.0f}%", fontsize=15,
                        ha='center', va='bottom', color=warehouse_colors[0],
                        transform=ax.transAxes)
                ax.text(0.7, 1.05, f"{capacity_distribution[1]/(capacity_distribution[0] + capacity_distribution[1]) * 100:.0f}%", fontsize=15,
                        ha='center', va='bottom', color=warehouse_colors[1],
                        transform=ax.transAxes)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.15)

    plt.savefig('visualization_examples/myopic_vs_lookahead.pdf', bbox_inches='tight', dpi=600)
    plt.show()



if __name__ == "__main__":

    
    visualize_assignments(
        warehouses_location=[[-50, -50], [50, 50]],
        all_warehouses=[0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        all_customers=[[0, -25, -25], [1, 25, 25], [2, -75, -75], [3, 75, 75], 
                    [4, 0, 0], [5, 50, -25], [6, -50, 50], [7, 25, 75],
                        [8, -25, 25], [9, 0, -50], [10, 50, 0], [11, -50, 0], 
                        [12, 0, 50], [13, -75, 0], [14, 75, 0], [15, 0, -75]]
    )

    # Call the function to generate the warehouse layout visualization
    generate_warehouse_visualization()

    # Call the function to generate the myopic vs lookahead visualization
    generate_myopic_vs_lookahead_visualization()

    #Get gif

    from env import InventoryEnv
    from miscelaneous import distance_calculator

    def policy(state):
                    
        # Select the closest warehouse among those with capacity > 0
        valid_warehouses = [
            i for i in range(state['static_info']['num_warehouses'])
            if state['warehouses_capacity'][i] > 0
        ]

        distance_to_warehouses = {
            i: distance_calculator(
                state['new_customer'][1:3],
                state['static_info']['warehouses_location'][i]
            )
            for i in valid_warehouses
        }
        min_distance = min(distance_to_warehouses.values())
        candidates = [i for i, distance in distance_to_warehouses.items() if distance == min_distance]
        
        action = candidates[0]
        
        return action

    env = InventoryEnv(num_warehouses=4, num_customers=50, capacity_distribution='uniform')

    def get_state(state):
        return state
    env.get_state = get_state

    done = False
    state, _ = env.reset()

    while not done:
        action = policy(state)
        state, reward, done, truncated, _ = env.step(action)

    visualize(env.warehouses_location, env.warehouses_initial_capacity, env.all_warehouses, env.all_customers, env.all_capacities)
