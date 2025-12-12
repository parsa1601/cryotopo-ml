import numpy as np
import time
from .utils import imsd, lp_code, dtw

class LPTDMethod:
    def __init__(self):
        pass

    def run(self, helices_datapoints, cryo_datapoints, classes, sticks, mode="Helix", run_dtw=False, ground_truth_mapping=None):
        """
        Runs the LPTD algorithm.
        
        Args:
            helices_datapoints: (N, 3) array of coordinates for helices/strands
            cryo_datapoints: (M, 3) array of coordinates for sticks
            classes: (N,) array of helix/strand IDs
            sticks: (M,) array of stick IDs
            mode: "Helix" or "Strand"
            run_dtw: bool, whether to run DTW for direction analysis
            ground_truth_mapping: dict or None, mapping helix/strand ID -> stick ID for accuracy calculation
            
        Returns:
            final_topology: List of dicts with keys 'num_helix'/'num_strand', 'num_stick', 'Direction'
            runtime: float (seconds)
        """
        start_time = time.time()
        
        # 1. Prepare Voxels
        # Group points by ID
        helix_ids = np.unique(classes)
        stick_ids = np.unique(sticks)
        
        helix_voxels = {}
        for hid in helix_ids:
            helix_voxels[hid] = helices_datapoints[classes == hid]
            
        stick_voxels = {}
        for sid in stick_ids:
            points = cryo_datapoints[sticks == sid]
            # For strands, the data loader already provides generated voxels from
            # the _Generated_sticks_strands.csv file (created by MATLAB's Bresenham).
            # So we DON'T need to run Bresenham again here!
            stick_voxels[sid] = points

        num_helices = len(helix_ids)
        num_sticks = len(stick_ids)
        
        # 2. IMSD and Weight Matrix
        weight_matrix = np.zeros((num_helices, num_sticks))
        
        # Map indices to IDs for matrix construction
        # Rows: 0..num_helices-1 corresponding to helix_ids[0]..helix_ids[-1]
        # Cols: 0..num_sticks-1 corresponding to stick_ids[0]..stick_ids[-1]

        for i in range(num_helices):
            hid = helix_ids[i]
            A = helix_voxels[hid]
            for j in range(num_sticks):
                sid = stick_ids[j]
                B = stick_voxels[sid]
                
                d = imsd(A, B)
                min_dist = np.min(d)
                if min_dist > 1e5:
                    min_dist = 1e5
                
                weight_matrix[i, j] = min_dist
                
        # 3. Iterative LP
        # We need to replicate the loop: for k=1:helix_number
        # Remove stick 1 (column 0) and helix k (row k)
        lp_reduced_results = []

        # Base LP (just to have it, though loop does the work)
        # f_base = lp_code(weight_matrix)
        
        for k in range(num_helices):
            # Create reduced matrix
            # Remove column 0 (Stick 1)
            # Remove row k (Helix k)

            # Note: The MATLAB code assumes we are forcing Helix k to Stick 1.
            # Stick 1 is the first stick in our list (stick_ids[0]).
            
            if num_sticks < 1:
                break
                
            # Reduced matrix: remove row k, col 0
            reduced_weight_matrix = np.delete(weight_matrix, k, axis=0)
            reduced_weight_matrix = np.delete(reduced_weight_matrix, 0, axis=1)
            
            # Run LP
            permvec, fval = lp_code(reduced_weight_matrix)
            
            fval_sum = fval + weight_matrix[k, 0]
            
            # Reconstruct topology
            # permvec maps rows of reduced matrix to cols of reduced matrix
            # We need to map back to original indices
            
            # Original row indices (excluding k)
            orig_rows = list(range(num_helices))
            orig_rows.pop(k)
            
            # Original col indices (excluding 0)
            orig_cols = list(range(num_sticks))
            orig_cols.pop(0)
            
            current_topology = []
            
            # Add the forced assignment: Helix k -> Stick 0
            current_topology.append({
                'num_helix' if mode == "Helix" else 'num_strand': helix_ids[k],
                'num_stick': stick_ids[0]
            })
            
            # Add assignments from LP
            for r_idx, c_idx in enumerate(permvec):
                if c_idx != -1: # If assigned
                    # r_idx is index in reduced matrix -> orig_rows[r_idx]
                    # c_idx is index in reduced matrix -> orig_cols[c_idx]
                    
                    h_id = helix_ids[orig_rows[r_idx]]
                    s_id = stick_ids[orig_cols[c_idx]]
                    
                    current_topology.append({
                        'num_helix' if mode == "Helix" else 'num_strand': h_id,
                        'num_stick': s_id
                    })
                else:
                    # Unassigned helix
                    h_id = helix_ids[orig_rows[r_idx]]
                    current_topology.append({
                        'num_helix' if mode == "Helix" else 'num_strand': h_id,
                        'num_stick': 0 # 0 means no match
                    })
            
            # Sort topology by helix ID for consistency
            current_topology.sort(key=lambda x: x['num_helix' if mode == "Helix" else 'num_strand'])
            
            # Calculate topology accuracy if ground truth is provided
            topology_accuracy = 0.0
            if ground_truth_mapping is not None:
                count_correct = 0
                for item in current_topology:
                    h_id = item['num_helix' if mode == "Helix" else 'num_strand']
                    s_id = item['num_stick']
                    if h_id in ground_truth_mapping and ground_truth_mapping[h_id] == s_id:
                        count_correct += 1
                topology_accuracy = (count_correct / len(current_topology)) * 100 if len(current_topology) > 0 else 0.0
            
            lp_reduced_results.append({
                'k': k,
                'lp_topology': current_topology,
                'lp_score': fval_sum,
                'topology_accuracy': topology_accuracy
            })
            
        # 4. Select Best Topology
        # First sort by lp_score (as in MATLAB)
        lp_reduced_results.sort(key=lambda x: x['lp_score'])
        
        # If ground truth is available, select by maximum accuracy (as in MATLAB)
        # Otherwise, select by minimum score
        if ground_truth_mapping is not None:
            best_result = max(lp_reduced_results, key=lambda x: x['topology_accuracy'])
        else:
            best_result = lp_reduced_results[0]
            
        final_topology = best_result['lp_topology']
        
        # 5. Direction Finding (DDA)
        if run_dtw:
            for item in final_topology:
                hid = item['num_helix' if mode == "Helix" else 'num_strand']
                sid = item['num_stick']
                
                if sid != 0:
                    A = helix_voxels[hid]
                    B1 = stick_voxels[sid]
                    
                    # Inverse B1
                    B2 = B1[::-1]
                    
                    # DTW
                    # A is (N, 3), B1 is (M, 3)
                    # MATLAB: dtw_x1=dtw(A(:,1),B1(:,1)); ... sum
                    
                    dtw_x1 = dtw(A[:, 0], B1[:, 0])
                    dtw_y1 = dtw(A[:, 1], B1[:, 1])
                    dtw_z1 = dtw(A[:, 2], B1[:, 2])
                    dist1 = dtw_x1 + dtw_y1 + dtw_z1
                    
                    dtw_x2 = dtw(A[:, 0], B2[:, 0])
                    dtw_y2 = dtw(A[:, 1], B2[:, 1])
                    dtw_z2 = dtw(A[:, 2], B2[:, 2])
                    dist2 = dtw_x2 + dtw_y2 + dtw_z2
                    
                    if dist1 <= dist2:
                        if mode == "Helix":
                            direction = 1 if dist1 < dist2 else -1
                        else:
                            direction = 1 if dist1 <= dist2 else -1
                    else:
                        direction = -1
                else:
                    direction = 0
                    
                item['Direction'] = direction
            
        end_time = time.time()
        runtime = end_time - start_time
        
        return final_topology, runtime

