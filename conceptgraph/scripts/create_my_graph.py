import gzip
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import networkx as nx
import imageio

def create_graph_from_data(classes, image_crops, xyxy, class_id, image_height, image_width):
    """
    Creates a graph representing spatial relationships between objects in images.
    Includes additional spatial relationships: 'part_of', 'overlapping',
    'to_the_left_of', and 'to_the_right_of'.

    Args:
        classes: List of class names.
        image_crops: List of PIL Image objects.
        xyxy: NumPy array of bounding box coordinates (x1, y1, x2, y2) for each object.
        class_id: NumPy array of class IDs corresponding to each bounding box.
        image_height: Height of the original image.
        image_width: Width of the original image.

    Returns:
        A networkx.Graph object representing the spatial relationships.
    """

    G = nx.Graph()

    encountered_classes = []
    object_instances = {}

    # Add nodes for each object instance
    for i in range(len(class_id)):
        class_name = classes[class_id[i]]

        if class_name not in encountered_classes:
            encountered_classes.append(class_name)
            object_instances[class_name] = 1
        else:
            object_instances[class_name] = object_instances.get(class_name, 0) + 1
        
        if sum([class_id[i] == class_id[k] for k in range(len(class_id))]) > 1:
            # If there are multiple instances of the same class, use a unique identifier
            node_name = f"{class_name}_{object_instances[class_name]}"
        else:
            # If only one instance, use the class name
            node_name = class_name

        node_name = node_name# Unique node name
        G.add_node(node_name,
                   class_name=class_name,
                   xyxy=xyxy[i],
                   image_crop_index=i)  # Store index to access image_crops

    def get_bbox_coords(node):
        return G.nodes[node]['xyxy']

    def calculate_overlap(box1, box2):
        """Calculates the overlap area between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap
        return (x_right - x_left) * (y_bottom - y_top)

    def is_above(box1, box2, threshold_y=0.2):
        """Checks if box1 is above box2."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        # Check if bottom of box1 is above top of box2 and there's some horizontal overlap
        return y2_1 < y1_2 and max(x1_1, x1_2) < min(x2_1, x2_2)

    def is_below(box1, box2, threshold_y=0.2):
        """Checks if box1 is below box2."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        # Check if top of box1 is below bottom of box2 and there's some horizontal overlap
        return y1_1 > y2_2 and max(x1_1, x1_2) < min(x2_1, x2_2)

    def is_next_to(box1, box2, threshold_x=0.3):
        """Checks if box1 is next to box2 (horizontally)."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        # Check if right of box1 is to the left of left of box2 OR vice versa, with some vertical overlap
        return (x2_1 < x1_2 or x1_1 > x2_2) and max(y1_1, y1_2) < min(y2_1, y2_2)

    def is_part_of(inner_box, outer_box, overlap_threshold=0.8):
        """Checks if inner_box is largely part of outer_box."""
        x1_i, y1_i, x2_i, y2_i = inner_box
        x1_o, y1_o, x2_o, y2_o = outer_box

        # Check if inner box is contained within the outer box
        if x1_i >= x1_o and y1_i >= y1_o and x2_i <= x2_o and y2_i <= y2_o:
            overlap_area = calculate_overlap(inner_box, outer_box)
            inner_area = (x2_i - x1_i) * (y2_i - y1_i)
            if inner_area > 0 and overlap_area / inner_area > overlap_threshold:
                return True
        return False

    def is_overlapping(box1, box2):
        """Checks if two boxes overlap (but not necessarily one fully on top)."""
        overlap = calculate_overlap(box1, box2)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        # Consider it overlapping if there's a significant intersection but not a high "part_of" ratio
        if overlap > 0 and not (overlap / area1 > 0.7 or overlap / area2 > 0.7):
            return True
        return False

    def is_to_the_left_of(box1, box2, threshold_x=0.1):
        """Checks if the center of box1 is significantly to the left of the center of box2."""
        center_x1 = (box1[0] + box1[2]) / 2
        center_x2 = (box2[0] + box2[2]) / 2
        return center_x1 < center_x2 and abs(center_x1 - center_x2) > threshold_x * image_width

    def is_to_the_right_of(box1, box2, threshold_x=0.1):
        """Checks if the center of box1 is significantly to the right of the center of box2."""
        center_x1 = (box1[0] + box1[2]) / 2
        center_x2 = (box2[0] + box2[2]) / 2
        return center_x1 > center_x2 and abs(center_x1 - center_x2) > threshold_x * image_width

    # Add edges based on spatial relationships
    nodes = list(G.nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1 = nodes[i]
            node2 = nodes[j]
            box1 = get_bbox_coords(node1)
            box2 = get_bbox_coords(node2)

            if is_part_of(box1, box2):
                G.add_edge(node1, node2, relation="part_of")
            elif is_part_of(box2, box1):
                G.add_edge(node2, node1, relation="part_of") # Edge direction might matter for 'part_of'
            elif calculate_overlap(box1, box2) > 0 and not is_part_of(box1, box2) and not is_part_of(box2, box1):
                # Prioritize 'on_top_of' if one is significantly higher
                if is_above(box1, box2, threshold_y=0.1):
                    G.add_edge(node1, node2, relation="on_top_of")
                elif is_above(box2, box1, threshold_y=0.1):
                    G.add_edge(node2, node1, relation="on_top_of")
                elif is_overlapping(box1, box2):
                    G.add_edge(node1, node2, relation="overlapping")
            elif is_above(box1, box2):
                G.add_edge(node1, node2, relation="above")
            elif is_below(box1, box2):
                G.add_edge(node1, node2, relation="below")
            elif is_next_to(box1, box2):
                G.add_edge(node1, node2, relation="next_to")
            elif is_to_the_left_of(box1, box2):
                G.add_edge(node1, node2, relation="to_the_left_of")
            elif is_to_the_right_of(box1, box2):
                G.add_edge(node1, node2, relation="to_the_right_of")

    return G

def visualize_graph(G, image_crops):
    """
    Visualizes the graph with bounding boxes and relationships.

    Args:
        G: The networkx.Graph object.
        image_crops: List of PIL Image objects.
    """

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111) 

    pos = nx.spring_layout(G, k=0.8)  # Adjust layout as needed

    # Draw nodes
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10)

    # Draw edges with labels
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Optionally display image crops (simplified for demonstration)
    # This part needs more sophisticated placement logic if you want to
    # embed the crops directly into the graph visualization.
    # for i, crop in enumerate(image_crops):
    #     plt.imshow(crop, extent=(i * 1.2, -1, i * 1.2 + 1, 0)) # Basic placement
    # for n, (x, y) in pos.items():
    #     index = list(G.nodes).index(n)
    #     if index < len(image_crops):
    #         try:
    #             bbox = G.nodes[n]['xyxy']
    #             width = bbox[2] - bbox[0]
    #             height = bbox[3] - bbox[1]
    #             aspect = width / height if height > 0 else 1
    #             size = 0.2
    #             crop_width = size
    #             crop_height = size / aspect
    #             plt.imshow(image_crops[index], extent=(x - crop_width / 2, x + crop_width / 2, y - crop_height / 2, y + crop_height / 2), aspect='auto')
    #         except Exception as e:
    #             print(f"Error displaying crop for node {n}: {e}")
    
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)  

    return img_array

def animate(i, *args):
    # graph_sequence = args[0]
    plt.clf()  # Clear the previous frame
    G = args[i]

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.8)

    # Draw nodes
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10)

    # Draw edges with labels
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(f"Frame {i+1}")
    return plt.gcf(), # Return the figure

# Function to get center of bounding box
def get_box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate areas
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Calculate distance between boxes
def calculate_distance(box1, box2):
    center1 = get_box_center(box1)
    center2 = get_box_center(box2)
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


def add_nodes(data):
    G = nx.Graph()

    xyxy = data['xyxy']
    class_ids = data['class_id']
    classes = data['classes']


    # Add nodes
    for i, (box, class_id) in enumerate(zip(xyxy, class_ids)):
        class_name = classes[class_id]
        center = get_box_center(box)
        
        # Use both index and class name to create unique node IDs for multiple instances
        node_id = f"{class_name}_{i}"
        G.add_node(node_id, 
                pos=center, 
                class_name=class_name, 
                box=box,
                class_id=class_id)

    # Connect nodes based on spatial relationships
    # We'll connect if boxes are close or overlapping
    threshold_distance = 300  # Adjust based on your image size

    for i, node1 in enumerate(G.nodes):
        box1 = G.nodes[node1]['box']
        for j, node2 in enumerate(G.nodes):
            if i < j:  # Avoid duplicate edges and self-loops
                box2 = G.nodes[node2]['box']
                distance = calculate_distance(box1, box2)
                iou = calculate_iou(box1, box2)
                
                # Connect if boxes are close enough or overlapping
                if distance < threshold_distance or iou > 0:
                    # Edge weight can be inverse of distance or based on IoU
                    weight = 1.0 / (distance + 1) if distance > 0 else 1.0
                    G.add_edge(node1, node2, weight=weight, distance=distance, iou=iou)

    # Get positions for visualization
    pos = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(12, 10))

    # Draw original image boundaries
    img_width, img_height = 640, 480
    plt.plot([0, img_width, img_width, 0, 0], [0, 0, img_height, img_height, 0], 'k--', alpha=0.3)

    # Draw the bounding boxes
    for node, attr in G.nodes(data=True):
        x1, y1, x2, y2 = attr['box']
        width = x2 - x1
        height = y2 - y1
        plt.gca().add_patch(Rectangle((x1, y1), width, height, 
                                    fill=False, edgecolor='green', alpha=0.5))

    # Draw the graph
    node_colors = [G.nodes[node]['class_id'] for node in G.nodes()]
    labels = {node: G.nodes[node]['class_name'] for node in G.nodes()}

    nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors, 
            node_size=1000, cmap=plt.cm.tab10, font_size=8, font_weight='bold')

    # Draw edges with width proportional to weight
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7)

    plt.title('Spatial Relationship Graph of Detected Objects')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Print edge information
    print("Edge information:")
    for u, v, data in G.edges(data=True):
        print(f"{u} - {v}: distance={data['distance']:.2f}, IoU={data['iou']:.2f}")


def main():
    pickle_path = "/scratch3/kat049/concept-graphs/my_local_data/DARPA/gsa_detections_ram_withbg_allclasses"
    files = sorted([f for f in os.listdir(pickle_path)])
    frames  = []
    graph_sequence = []
    for file in files:
        if file != 'frame_1461.pkl.gz':
            continue
        with gzip.open(os.path.join(pickle_path, file), 'rb') as f:
            data = pickle.load(f)
            G = create_graph_from_data(data['classes'], data['image_crops'], data['xyxy'], data['class_id'], 480, 640)
            frames.append(visualize_graph(G, data['image_crops']))
    #         graph_sequence.append(G)
    # fig = plt.figure(figsize=(12, 12))
    # ani = animation.FuncAnimation(fig, animate, fargs=(graph_sequence),
    #                           frames=len(graph_sequence), interval=500, blit=False, repeat=False)

    # ani.save('graph_animation.mp4', writer='ffmpeg', fps=2)
    imageio.mimsave('graph_animation.mp4', frames, fps=10) 
if __name__ == "__main__":
    main()