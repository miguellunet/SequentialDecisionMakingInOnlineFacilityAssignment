
import string
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np

def visualize(warehouses_location, warehouses_initial_capacity, all_warehouses, all_customers, all_capacities):

    warehouse_colors = ['#2b83ba','#d7191c', '#abdda4', '#fdae61','#ffffbf']
    
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
    
    warehouse_colors = ['#2b83ba', '#d7191c', '#abdda4', '#fdae61','#ffffbf']

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

    warehouse_colors = ['#2b83ba', '#d7191c', '#abdda4', '#fdae61','#ffffbf']
    num_warehouses_list = [2, 3, 4, 5]

    for row, distribution_type in enumerate(["uniform", "uneven"]):
        for col, num_warehouses in enumerate(num_warehouses_list):
            ax = axes[row, col]
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            ax.set_aspect('equal', adjustable='box')
            if row == 0:
                ax.set_title(f"{num_warehouses} facilities", fontsize=12)
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
                ax.add_patch(patches.Circle(loc, 11, fill=True, facecolor=warehouse_colors[i % len(warehouse_colors)], edgecolor='black', linewidth=1.2, zorder=2))
                ax.text(loc[0], loc[1], facility_letters[i % len(facility_letters)], ha='center', va='center', fontsize=11, color='black', zorder=3) #fontweight='bold'
                ax.text(loc[0], loc[1] + 15, f"{cap}%", ha='center', fontsize=12, color='black')

        # Add distribution type text on the left side
        ax = axes[row, 0]
        ax.text(-130, 0, f"{distribution_type.capitalize()} distribution", fontsize=12, ha='center', va='center', rotation=90)

    plt.tight_layout()
    plt.savefig('visualization_examples/facilities.pdf', dpi=600)
    plt.show()


def generate_myopic_vs_lookahead_visualization():
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), gridspec_kw={'width_ratios': [0.2, 1, 1, 1], 'height_ratios': [0.2, 1, 1]})  # 3 rows, 4 columns with adjusted ratios
    for ax in axes[0, :]:
        ax.axis('off')  # Turn off the top row
    for ax in axes[:, 0]:
        ax.axis('off')  # Turn off the leftmost column

    warehouse_colors = ['#2b83ba','#d7191c', '#abdda4', '#fdae61','#ffffbf']
    capacity_distribution_settings = [[50,50], [80,20], [20,80]]

    def get_greedy_action(state):
        #Calculate distance to 0 and to 1
        distance_to_0 = np.sqrt((state[0] + 50) ** 2 + (state[1] + 50) ** 2)
        distance_to_1 = np.sqrt((state[0] - 50) ** 2 + (state[1] - 50) ** 2)
        #If the distance to 0 is smaller than the distance to 1, return 0, else return 1
        if distance_to_0 < distance_to_1:
            return 0
        else:
            return 1
    
    def get_lookahead_action(state):
        #Calculate distance to 0 and to 1
        distance_to_0 = np.sqrt((state[0] + 50) ** 2 + (state[1] + 50) ** 2)
        distance_to_1 = np.sqrt((state[0] - 50) ** 2 + (state[1] - 50) ** 2)
        #If the distance to 0 is smaller than the distance to 1, return 0, else return 1
        if distance_to_0-1*state[2] < distance_to_1-1*state[3]:
            return 0
        else:
            return 1

    for row, distribution_type in enumerate(["myopic", "lookahead"]):
        
        #Print "greedy" and "lookahead" in the leftmost column
        ax = axes[row+1, 0]
        ax.text(0.5, 0.55, f"{distribution_type.capitalize()}", fontsize=15, ha='center')
        ax.text(0.5, 0.45, "policy", fontsize=15, ha='center')

        for col, capacity_distribution in enumerate(capacity_distribution_settings):
            
            #Write the capacity distribution in the top row
            ax = axes[0, col+1]
            # Write "Capacity Distribution" in the top row
            ax.text(0.5, 0.5, "Capacity distribution", fontsize=15, ha='center')
            ax.text(0.3, 0.2, f"{capacity_distribution[0]}%", fontsize=15, ha='center', color=warehouse_colors[0])
            ax.text(0.7, 0.2, f"{capacity_distribution[1]}%", fontsize=15, ha='center', color=warehouse_colors[1])

            ax = axes[row+1, col+1]
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            #DONT PLOT THE x and y axis
            ax.axis('off')
            #ax.set_title(f"{num_warehouses} warehouses - {distribution_type} capacity distribution")
            #ax.set_xticks(np.arange(-100, 101, 50))
            #ax.set_yticks(np.arange(-100, 101, 50))
            #ax.grid(True, linestyle='--', linewidth=0.5)
            square = 10
            for i in range(-100+int(square/2), 101-int(square/2), int(square)):
                for j in range(-100+int(square/2), 101-int(square/2), int(square)):
                    my_state = (i, j) + tuple(cap for cap in capacity_distribution)
                    if distribution_type == "myopic":
                        color = warehouse_colors[get_greedy_action(my_state)]
                    else:
                        color = warehouse_colors[get_lookahead_action(my_state)]

                    ax.add_patch(plt.Rectangle((i-square/2, j-square/2), square, square, fill=True, color=color, alpha=0.3))#
            warehouses_location = [[-50, -50], [50, 50]]

            for i, loc in enumerate(warehouses_location):
                ax.add_patch(patches.Circle(loc, 12, fill=True, color=warehouse_colors[i % len(warehouse_colors)]))
                warehouse_name = f"{chr(65 + i)}"
                ax.text(loc[0], loc[1]-3, warehouse_name, ha='center', fontsize=16, color='white', fontweight='bold')


            #Put legend indicating x coordinate and y coordinate
            #ax.text(0.5, -0.1, "X coordinate", fontsize=12, ha='center', va='center', transform=ax.transAxes)
            #ax.text(-0.1, 0.5, "Y coordinate", fontsize=12, ha='center', va='center', rotation=90, transform=ax.transAxes)

    plt.tight_layout()

    #Save the figure
    plt.savefig('visualization_examples/myopic_vs_lookahead.pdf', dpi=600)
    plt.show()


def generate_dla_visualization():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 2 rows, 3 columns

    warehouse_colors = ['#2b83ba','#d7191c', '#abdda4', '#fdae61','#ffffbf']
    num_warehouses_list = [2, 3, 4]

    ax = axes[0]
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_title("Before order arrival\n(16 artificial orders)")
    ax.set_xticks(np.arange(-100, 101, 50))
    ax.set_yticks(np.arange(-100, 101, 50))
    #ax.grid(True, linestyle='--', linewidth=0.5)

    warehouses_location = [[-50, -50], [50, 50]]
    capacities = [50, 50]

    # Plot warehouses
    for i, (loc, cap) in enumerate(zip(warehouses_location, capacities)):
        ax.add_patch(patches.Circle(loc, 10, fill=True, color=warehouse_colors[i % len(warehouse_colors)]))
        ax.text(loc[0], loc[1]-3, f"{chr(65 + i)}", ha='center', fontsize=15, color='white', fontweight='bold')

    for i in range(-75, 76, 50):
        for j in range(-75, 76, 50):
            ax.add_patch(patches.Circle((i, j), 2, fill=True, color='black'))

    ax = axes[1]
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_title("After order arrival\n(15 artificial and 1 real order)")
    ax.set_xticks(np.arange(-100, 101, 50))
    ax.set_yticks(np.arange(-100, 101, 50))
    #ax.grid(True, linestyle='--', linewidth=0.5)

    # Plot warehouses
    for i, (loc, cap) in enumerate(zip(warehouses_location, capacities)):
        ax.add_patch(patches.Circle(loc, 10, fill=True, color=warehouse_colors[i % len(warehouse_colors)]))
        ax.text(loc[0], loc[1]-3, f"{chr(65 + i)}", ha='center', fontsize=15, color='white', fontweight='bold')

    for i in range(-75, 76, 50):
        for j in range(-75, 76, 50):
            if not (i == 25 and j == 75):
                ax.add_patch(patches.Circle((i, j), 2, fill=True, color='black'))

    # Plot a star in coordinates (50,60)
    ax.add_patch(patches.Rectangle((35 - 5, 70 - 5), 10, 10, fill=True, color='black'))

    ax = axes[2]
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_title("Order-facility assignment")
    ax.set_xticks(np.arange(-100, 101, 50))
    ax.set_yticks(np.arange(-100, 101, 50))
    #ax.grid(True, linestyle='--', linewidth=0.5)

    # Plot warehouses
    for i, (loc, cap) in enumerate(zip(warehouses_location, capacities)):
        ax.add_patch(patches.Circle(loc, 10, fill=True, color=warehouse_colors[i % len(warehouse_colors)]))
        ax.text(loc[0], loc[1]-3, f"{chr(65 + i)}", ha='center', fontsize=15, color='white', fontweight='bold')

    for i in range(-75, 76, 50):
        for j in range(-75, 76, 50):
            if not (i == 25 and j == 75):
                ax.add_patch(patches.Circle((i, j), 2, fill=True, color='black'))
                if j > 0:
                    # Create a red line to the first warehouse
                    ax.plot([i, warehouses_location[1][0]], [j, warehouses_location[1][1]], color=warehouse_colors[1], linestyle='--')
                else:
                    ax.plot([i, warehouses_location[0][0]], [j, warehouses_location[0][1]], color=warehouse_colors[0], linestyle='--')

    # Plot a star in coordinates (50,60)
    ax.add_patch(patches.Rectangle((35 - 5, 70 - 5), 10, 10, fill=True, color='black'))
    ax.plot([35, warehouses_location[1][0]], [70, warehouses_location[1][1]], color=warehouse_colors[1], linestyle='--')

    plt.tight_layout()
    plt.savefig('visualization_examples/dla_example.pdf', dpi=600)
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

    # Call the function to generate the DLA visualization
    generate_dla_visualization()
    


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
