import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.lines import Line2D

def draw_pitch(ax, length=105, width=68):
    x_min, x_max = -length/2, length/2
    y_min, y_max = -width/2, width/2
    
    ax.plot([x_min, x_max], [y_min, y_min], color='black', linewidth=1)
    ax.plot([x_min, x_max], [y_max, y_max], color='black', linewidth=1)
    ax.plot([x_min, x_min], [y_min, y_max], color='black', linewidth=1)
    ax.plot([x_max, x_max], [y_min, y_max], color='black', linewidth=1)
    ax.plot([0, 0], [y_min, y_max], color='black', linewidth=1)
    
    center_circle = patches.Circle((0, 0), radius=9.15, fill=False, color='black', linewidth=1)
    ax.add_patch(center_circle)
    ax.scatter(0, 0, color='black', s=10) # Center point
    
    penalty_area_width = 40.32
    penalty_area_length = 16.5
    goal_area_width = 18.32
    goal_area_length = 5.5
    
    # Left Penalty Area
    ax.plot([x_min, x_min + penalty_area_length], [penalty_area_width/2, penalty_area_width/2], color='black', linewidth=1)
    ax.plot([x_min, x_min + penalty_area_length], [-penalty_area_width/2, -penalty_area_width/2], color='black', linewidth=1)
    ax.plot([x_min + penalty_area_length, x_min + penalty_area_length], [-penalty_area_width/2, penalty_area_width/2], color='black', linewidth=1)

    # Right Penalty Area
    ax.plot([x_max, x_max - penalty_area_length], [penalty_area_width/2, penalty_area_width/2], color='black', linewidth=1)
    ax.plot([x_max, x_max - penalty_area_length], [-penalty_area_width/2, -penalty_area_width/2], color='black', linewidth=1)
    ax.plot([x_max - penalty_area_length, x_max - penalty_area_length], [-penalty_area_width/2, penalty_area_width/2], color='black', linewidth=1)

    # Left Goal Area
    ax.plot([x_min, x_min + goal_area_length], [goal_area_width/2, goal_area_width/2], color='black', linewidth=1)
    ax.plot([x_min, x_min + goal_area_length], [-goal_area_width/2, -goal_area_width/2], color='black', linewidth=1)
    ax.plot([x_min + goal_area_length, x_min + goal_area_length], [-goal_area_width/2, goal_area_width/2], color='black', linewidth=1)

    # Right Goal Area
    ax.plot([x_max, x_max - goal_area_length], [goal_area_width/2, goal_area_width/2], color='black', linewidth=1)
    ax.plot([x_max, x_max - goal_area_length], [-goal_area_width/2, -goal_area_width/2], color='black', linewidth=1)
    ax.plot([x_max - goal_area_length, x_max - goal_area_length], [-goal_area_width/2, goal_area_width/2], color='black', linewidth=1)

    # Goals (outside pitch)
    goal_width = 7.32
    # Left Goal
    ax.plot([x_min, x_min - 2], [goal_width/2, goal_width/2], color='black', linewidth=1)
    ax.plot([x_min, x_min - 2], [-goal_width/2, -goal_width/2], color='black', linewidth=1)
    ax.plot([x_min - 2, x_min - 2], [-goal_width/2, goal_width/2], color='black', linewidth=1)
    
    # Right Goal
    ax.plot([x_max, x_max + 2], [goal_width/2, goal_width/2], color='black', linewidth=1)
    ax.plot([x_max, x_max + 2], [-goal_width/2, -goal_width/2], color='black', linewidth=1)
    ax.plot([x_max + 2, x_max + 2], [-goal_width/2, goal_width/2], color='black', linewidth=1)
    
    # Penalty Arcs (Approximate)
    # Left
    left_arc = patches.Arc((x_min + 11, 0), width=18.3, height=18.3, angle=0, theta1=-53, theta2=53, color='black')
    ax.add_patch(left_arc)
    # Right
    right_arc = patches.Arc((x_max - 11, 0), width=18.3, height=18.3, angle=0, theta1=127, theta2=233, color='black')
    ax.add_patch(right_arc)
    
    # Penalty Spots
    ax.scatter(x_min + 11, 0, color='black', s=5)
    ax.scatter(x_max - 11, 0, color='black', s=5)
    
    return ax

def plot_scene(ax, scene_tensor, title=None):
    draw_pitch(ax)
    
    if scene_tensor is None:
        ax.axis('off')
        return

    n_players, _, n_frames = scene_tensor.shape
    
    team1_indices = range(0, 11)
    team2_indices = range(11, 22)
    
    # Plot Team 1 (Blue)
    for i in team1_indices:
        if i >= n_players:
            break
        track = scene_tensor[i] # (2, T)
        xs = track[0]
        ys = track[1]
        
        # Filter NaNs
        mask = ~np.isnan(xs)
        xs = xs[mask]
        ys = ys[mask]
        
        if len(xs) == 0:
            continue
            
        # Plot trajectory line
        ax.plot(xs, ys, color='blue', alpha=0.6, linewidth=1.5)
        
        # Plot end position (current frame)
        ax.scatter(xs[-1], ys[-1], color='blue', s=80, alpha=0.8, edgecolors='white', zorder=5)
    
    # Plot Team 2 (Green)
    for i in team2_indices:
        if i >= n_players:
            break
        track = scene_tensor[i] # (2, T)
        xs = track[0]
        ys = track[1]
        
        # Filter NaNs
        mask = ~np.isnan(xs)
        xs = xs[mask]
        ys = ys[mask]
        
        if len(xs) == 0:
            continue
            
        # Plot trajectory line
        ax.plot(xs, ys, color='green', alpha=0.6, linewidth=1.5)
        
        # Plot end position (current frame)
        ax.scatter(xs[-1], ys[-1], color='green', s=80, alpha=0.8, edgecolors='white', zorder=5)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Team 1'),
        Line2D([0], [0], color='green', lw=2, label='Team 2')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
    if title:
        ax.set_title(title, fontsize=10)

    ax.set_aspect('equal')
    ax.set_xlim(-55, 55)
    ax.set_ylim(-36, 36)
    ax.axis('off')

def save_search_results(query_scene, results, output_file="search_results.png"):
    n_results = len(results)
    
    # Layout:
    # Row 1: Query (in center)
    # Row 2: Matches
    
    cols = 3
    rows = (n_results // cols) + 2 # +1 for query row, +partial
    
    if n_results <= 3:
        rows = 2
    
    fig = plt.figure(figsize=(15, 6 * rows))
    
    # Plot Query Scene (Spanning full width or centered)
    ax_q = plt.subplot2grid((rows, cols), (0, 0), colspan=cols)
    
    # Build query title with gameClock if available
    query_title = f"QUERY SCENE\nType: {query_scene['event_metadata'].get('type')} | ID: {query_scene['event_metadata'].get('game_event_id')}"
    if query_scene['event_metadata'].get('gameClock'):
        query_title += f" | Clock: {query_scene['event_metadata'].get('gameClock')}"
    
    plot_scene(ax_q, query_scene['scene_tensor'], title=query_title)
    
    # Plot Matches
    for i, res in enumerate(results):
        r = 1 + (i // cols)
        c = i % cols
        
        if r >= rows: break
        
        ax = plt.subplot2grid((rows, cols), (r, c))
        
        tensor = res.get('scene_tensor')
        dist = res.get('distance', 0)
        meta = res.get('metadata', {})
        
        plot_scene(ax, tensor, 
                   title=f"Rank {i+1}\nDist: {dist:.4f} | Type: {meta.get('type')}\nID: {meta.get('game_event_id')}")
        
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")

def plot_scene_raw(ax, raw_scene, title=None):
    draw_pitch(ax)
    
    if raw_scene is None:
        ax.text(0, 0, 'Scene not available', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return
    
    # Plot home team (blue)
    for jersey, coords in raw_scene['home'].items():
        if not coords:
            continue
        xs, ys = zip(*coords)
        ax.plot(xs, ys, color='blue', alpha=0.6, linewidth=1.5)
        ax.scatter(xs[-1], ys[-1], color='blue', s=80, alpha=0.8, edgecolors='white', zorder=5)
    
    # Plot away team (green)
    for jersey, coords in raw_scene['away'].items():
        if not coords:
            continue
        xs, ys = zip(*coords)
        ax.plot(xs, ys, color='green', alpha=0.6, linewidth=1.5)
        ax.scatter(xs[-1], ys[-1], color='green', s=80, alpha=0.8, edgecolors='white', zorder=5)
    
    # Plot ball (red dotted)
    if raw_scene['ball']:
        bxs, bys = zip(*raw_scene['ball'])
        ax.plot(bxs, bys, color='red', linestyle='dotted', linewidth=2, zorder=6)
        ax.scatter(bxs[-1], bys[-1], color='red', s=60, edgecolors='white', zorder=7)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Team 1 (Home)'),
        Line2D([0], [0], color='green', lw=2, label='Team 2 (Away)'),
        Line2D([0], [0], color='red', lw=2, linestyle='dotted', label='Ball')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')

    ax.set_aspect('equal')
    ax.set_xlim(-55, 55)
    ax.set_ylim(-36, 36)
    ax.axis('off')

def save_search_results_enhanced(query_scene, results, output_file="search_results_enhanced.png"):
    n_results = len(results)
    
    cols = 3
    rows = (n_results // cols) + 2
    
    if n_results <= 3:
        rows = 2
    
    fig = plt.figure(figsize=(15, 6 * rows))
    
    # Plot Query Scene
    ax_q = plt.subplot2grid((rows, cols), (0, 0), colspan=cols)
    
    # Build query title with gameClock if available
    query_title = f"QUERY SCENE\nType: {query_scene['event_metadata'].get('type')} | ID: {query_scene['event_metadata'].get('game_event_id')}"
    if query_scene['event_metadata'].get('gameClock'):
        query_title += f" | Clock: {query_scene['event_metadata'].get('gameClock')}"
    
    plot_scene_raw(ax_q, query_scene.get('raw_scene'), title=query_title)
    
    # Plot Matches
    for i, res in enumerate(results):
        r = 1 + (i // cols)
        c = i % cols
        
        if r >= rows: break
        
        ax = plt.subplot2grid((rows, cols), (r, c))
        
        raw = res.get('raw_scene')
        dist = res.get('distance', 0)
        meta = res.get('metadata', {})
        
        plot_scene_raw(ax, raw, 
                      title=f"Rank {i+1} | Dist: {dist:.4f}\nType: {meta.get('type')} | ID: {meta.get('game_event_id')}")
        
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
