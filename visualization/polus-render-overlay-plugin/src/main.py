import os, logging, argparse, json, math, re, itertools
import filepattern
from pathlib import Path
from utils import TextCell, GridCell, ChemCell, GridCellLayerSpec, TextLayerSpec, ChemLayerSpec, OverlaySpec, to_bijective
from pprint import pprint
import pandas as pd
from tqdm import tqdm
from parse import parse
from string import Formatter


# Import environment variables
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_EXT = os.environ.get("POLUS_EXT", ".ome.tif")

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)



def main(
    stitching_vector,
    outDir,
    file_pattern=None,
    heatmap=None,
    chem=None,
    text=False,
    concatenate=False
    ):

    #TODO: Get image size from meta data
    def get_var_stats(vp, img_size=(1080,1080)):

        var_order = []
        var_stats = {v:{} for v in vp.variables}
        
        # # Add the fov level grid size
        # var_stats['fov'] = {}
        # var_stats['fov']['x_size'] = img_size[1]
        # var_stats['fov']['y_size'] = img_size[0]
        
        
        # Get the top left grid (0,0) to determine the top left image
        for file in vp():
            if file[0]['gridX'] == '0' and file[0]['gridY'] == '0':
                for v in vp.variables:
                    var_stats[v]['top_left'] = file[0][v]
                break
        
        for v in vp.variables:

            files = [file for file in vp(group_by=v)]
            
            # Check if the variable is constant over the stitching vector if so
            # then it is a top grid level variable.
            if len(files[0]) < 2:
                x_grid = [int(file[0]['gridX']) for file in files]
                y_grid = [int(file[0]['gridY']) for file in files]
                
                # Get the variables pixel positions for one grid
                x_pos = [int(file[0]['posX']) for file in files]
                y_pos = [int(file[0]['posY']) for file in files]
                
                x_grid.sort()
                y_grid.sort()
                x_pos.sort()
                y_pos.sort()
                
                x_range = x_step = x_grid[-1] - x_grid[0]
                y_range = y_step = y_grid[-1] - y_grid[0]
                
            else:
                # Get the grid range for each variable which represents its
                # total range across its grid minus the size of its child grid
                x_grid = [int(file['gridX']) for file in files[0]]
                y_grid = [int(file['gridY']) for file in files[0]]
                
                # Get the variables pixel positions for one grid
                x_pos = [int(file['posX']) for file in files[0]]
                y_pos = [int(file['posY']) for file in files[0]]
                
                x_grid.sort()
                y_grid.sort()
                x_pos.sort()
                y_pos.sort()
                
                x_range = x_grid[-1] - x_grid[0]
                y_range = y_grid[-1] - y_grid[0]
                
                # Get the grid step for each variable; the step represents the 
                # distance to the next image when only the variable being 
                # considered is allowed to vary.
                x_step = x_grid[1] - x_grid[0]
                y_step = y_grid[1] - y_grid[0]
                
                
            var_stats[v]['x_range'] = x_range
            var_stats[v]['y_range'] = y_range
            var_stats[v]['x_step'] = x_step
            var_stats[v]['y_step'] = y_step
                
            # Save the variables pixel sizes for a single grid minus the
            # the size of the child grid
            var_stats[v]['x_size'] = x_pos[-1] - x_pos[0]
            var_stats[v]['y_size'] = y_pos[-1] - y_pos[0]
            
            # Save the variable, step size and x,y or w variable type. A x 
            # variable means the variable only vaires in the x dimension and y 
            # variable only varies in the y dimension. A w variable means it 
            # varies in both dimensions
            if y_range == 0:
                var_order.append([x_step, v, 'x', x_range])
                var_stats[v]['v_type'] = 'x'
                
            elif x_range == 0:
                var_order.append([y_step, v, 'y', y_range])
                var_stats[v]['v_type'] = 'y'
                
            else:
                step = max(x_step, y_step)
                var_order.append([step, v, 'w', x_range, y_range])
                var_stats[v]['v_type'] = 'w'
        
        # Sort from largest to smallest step size
        var_order.sort(reverse=True)
        
        logger.debug(var_stats)
        logger.debug(var_order)
        
        layout = ''
    
        # Define the current sub-grid length and height
        x_factor = 1
        y_factor = 1
        
        grid_dim = []
        
        # Define the child grize pixel size
        x_child = img_size[1]
        y_child = img_size[0]
        
        while var_order:
            
            # Define the current variable name, step and type
            v_data = var_order.pop()
            v = v_data[1]
            t = v_data[2]
            
            # Check if variable is a multi-dimensional variable 
            # (w stands for wrap)
            if t == 'w':
                
                x_range = var_stats[v]['x_range']
                y_range = var_stats[v]['y_range']
                
                # Set the variable's grid size as the child size and update 
                # child size for next iteration
                var_stats[v]['x_size'], x_child = x_child, var_stats[v]['x_size'] + x_child
                var_stats[v]['y_size'], y_child = y_child, var_stats[v]['y_size'] + y_child
                
                if var_order:
                    layout += v + ','
                    
                else:
                    layout += v
            
            elif t == 'x':
                
                s = v_data[0]/x_factor
                
                if var_order:
                    nv_data = var_order[-1]
                    nv = nv_data[1]
                    ns = nv_data[0]/y_factor
                    nt = nv_data[2]
                    
                    # Try to match the x variable with a y variable
                    if s == ns and nt == 'y':
                        layout += v + nv + ','
                        var_order.pop()
                        
                        x_range = var_stats[v]['x_range']
                        y_range = var_stats[nv]['y_range']
                        
                        var_stats[v]['x_size'], x_child = x_child, var_stats[v]['x_size'] + x_child
                        var_stats[nv]['y_size'], y_child = y_child, var_stats[nv]['y_size'] + y_child
                    
                    else:
                        layout += v + ','
                        x_range = var_stats[v]['x_range']
                        y_range = 0
                        
                        var_stats[v]['x_size'], x_child = x_child, var_stats[v]['x_size'] + x_child
                        var_stats[v]['y_size'] = y_child

                else:
                    layout += v  
                    x_range = var_stats[v]['x_range']
                    y_range = 0
                    
                    var_stats[v]['x_size'], x_child = x_child, var_stats[v]['x_size'] + x_child
                    var_stats[v]['y_size'] = y_child
                        
            elif t == 'y':
                
                s = v_data[0]/y_factor
                
                if var_order:
                    nv_data = var_order[-1]
                    nv = nv_data[1]
                    ns = nv_data[0]/x_factor
                    nt = nv_data[2]
                    
                    # Try to match the y variable with a x variable
                    if s == ns and var_stats[nv]['v_type'] == 'x':
                        layout += nv + v + ','
                        var_order.pop()
                        
                        x_range = var_stats[nv]['x_range']
                        y_range = var_stats[v]['y_range']
                        
                        var_stats[nv]['x_size'], x_child = x_child, var_stats[nv]['x_size'] + x_child
                        var_stats[v]['y_size'], y_child = y_child, var_stats[v]['y_size'] + y_child

                    else:
                        layout += v + ','
                        y_range = var_stats[v]['y_range']
                        x_range = 0
                        
                        var_stats[v]['y_size'], y_child = y_child, var_stats[v]['y_size'] + y_child
                        var_stats[v]['x_size'] = x_child

                else:
                    layout += v
                    y_range = var_stats[v]['y_range']
                    x_range = 0
                    
                    var_stats[v]['y_size'], y_child = y_child, var_stats[v]['y_size'] + y_child
                    var_stats[v]['x_size'] = x_child
                    
                    
            # The factors represent the number of images in the current sub-grid
            # for thier respective axis
            x_factor += x_range
            y_factor += y_range
            
            grid_dim.append((x_factor, y_factor))

        return var_stats, layout

    #TODO: Implement recursive solution which generates text overlays
    def generate_text_old(layout, grid, var_stats, vp) -> dict:
        """
        Given a specific layout and file list this function will generate a 
        text overlays for each grid level. One limitation of this approach is
        that if the top left image does not exist for a grid no overlay will
        be generated.
        """
        
        overlays = {level:[] for level in range(len(layout))}
        variables = {level:{} for level in range(len(layout))}
        levels = len(layout)
        
        variables[0]['pv'] = layout[0]
        
        # Define the variable and parent variable for each grid level
        for i, v in enumerate(layout[:-1]):
                
            level = i + 1
            variables[level]['v'] = v
            
            if len(v) == 2:
                variables[level]['vx'] = v[0]
                variables[level]['vy'] = v[1]
                
            # Get the parent grid variables
            pv = layout[i+1]
            variables[level]['pv'] = pv
            
            if len(pv) == 2:
                variables[level]['px'] = pv[0]
                variables[level]['py'] = pv[1]
        
        # Iterate over every file in the vector pattern
        for file in grid:
            
            # Skip images that are not in channel 1
            if file['c'] != 1:
                continue
            
            text = file['file']
            position = (file['posX'], file['posY'])
            
            # Add the base level (single image overlay)
            overlays[0].append(
                TextCell(position=position, text=text)
                )
            
            # For each grid level check if the image is a corner image
            for i in range(levels-1):
                
                level = i + 1
                
                # Define the current level's variables and parent variables
                v = variables[level]['v']
                pv = variables[level]['pv']
                    
                if len(v) == 2:
                    vx = variables[level]['vx']
                    vy = variables[level]['vy']
                    fx = file[vx]
                    fy = file[vy]

                    # Check if the image is top left for current grid level
                    if fx == var_stats[vx]['top_left'] and fy == var_stats[vy]['top_left']:
                        
                        # Get the parent grid data
                        if len(pv) == 2:
                            px = variables[level]['px']
                            py = variables[level]['py']
                            
                            # This ensures a range of 0 to x where x is the
                            # number of images in a row
                            fpx = abs(file[px] - var_stats[px]['top_left'])
                            
                            # This ensures a range of A to y where y is the
                            # number of images in a column (in bijective num.)
                            fpy = abs(file[py] - var_stats[py]['top_left']) + 1
                            
                            # Use bijective numeration for dual variable levels 
                            text = to_bijective(int(fpy)) + str(fpx)
                            overlays[level].append(
                                TextCell(position=position, text=text)
                                )
                        
                        else:
                            text = pv.upper() + str(file[pv])
                            overlays[level].append(
                                TextCell(position=position, text=text)
                                )
                        
                    else:
                        # If the sub-grid is not a corner image then it cannot 
                        # be a corner image for parent grids
                        break
                
                else:
                    fp = file[v]
                    
                    if fp == var_stats[v]['top_left']:
                        
                        # Get the parent grid data
                        if len(pv) == 2:
                            px = variables[level]['px']
                            py = variables[level]['py']
                            
                            # This ensures a range of 0 to x where x is the
                            # number of images in a row
                            fpx = abs(file[px] - var_stats[px]['top_left'])
                            
                            # This ensures a range of A to y where y is the
                            # number of images in a column (in bijective num.)
                            fpy = abs(file[py] - var_stats[py]['top_left']) + 1
                            
                            # Use bijective numeration for dual variable levels 
                            text = to_bijective(int(fpy)) + str(fpx)
                            overlays[level].append(
                                TextCell(position=position, text=text)
                                )
                        
                        else:
                            text = pv.upper() + str(file[pv])
                            overlays[level].append(
                                TextCell(
                                    position=position, 
                                    text=text
                                    )
                                )
                        
                    else:
                        # If the sub-grid is not a corner image then it cannot 
                        # be a corner image for parent grids
                        break
        
        
        text_layers = []
        
        for level, overlay in overlays.items():
            
            # For excluding text layers by grid level
            if level != 1:
                continue
            
            # Check if at top level
            if level == levels:
                v = variables[level]['v']
                width = 1
                height = 1
                
                if len(v) == 1:
                    cell_width = var_stats[v]['x_size']
                    cell_height = var_stats[v]['y_size']
                    
                else:
                    cell_width = var_stats[v[0]]['x_size']
                    cell_height = var_stats[v[1]]['y_size']
                    
            else:
                
                pv = variables[level]['pv']
            
                if len(pv) > 1:
                    width = len(vp.uniques[pv[0]])
                    height = len(vp.uniques[pv[1]])
                    cell_width = var_stats[pv[0]]['x_size']
                    cell_height = var_stats[pv[1]]['y_size']
                    
                else:
                    width = math.ceil(math.sqrt(len(vp.uniques[pv])))
                    height = math.floor(math.sqrt(len(vp.uniques[pv])))
                    cell_width = var_stats[pv]['x_size']
                    cell_height = var_stats[pv]['y_size']
            
            
            overlay = sorted(overlay)
            
            tls = TextLayerSpec(
                id=level,
                width=width,
                height=height,
                cell_size=cell_width,
                # cell_width=cell_width,
                # cell_height=cell_height,
                data=overlay
                )

            text_layers.append(tls)
        
        return text_layers
    
    
    def build_grids(vp, layout):
        
        grids = {}
        
        # Iterate over each variable group in layout
        for i, v in enumerate(layout):
            
            pv = layout[i+1]
            width = int(max(vp.uniques[pv[0]])) - int(min(vp.uniques[pv[0]])) + 1
            height = int(max(vp.uniques[pv[1]])) - int(min(vp.uniques[pv[1]])) + 1
            grids[i+1] = {'data':{}, 'parent':pv, 'width':width, 'height':height}
            
            
            
            # Group the grid level by the current variable group
            level_files = [g for g in vp(group_by=v) if g[0]['c'] != 2]

            # Iterate over each grid in the current level
            for grid_files in level_files:
                
                file_names = [file['file'] for file in grid_files]
                file_names.sort()
                file_key = ''.join(file_names)
                
                grids[i+1]['data'][file_key] = {}
                
                min_gridX = min([file['gridX'] for file in grid_files])
                max_gridX = max([file['gridX'] for file in grid_files])
                min_gridY = min([file['gridY'] for file in grid_files])
                max_gridY = max([file['gridY'] for file in grid_files])
                
                min_posX = min([file['posX'] for file in grid_files])
                max_posX = max([file['posX'] for file in grid_files])
                min_posY = min([file['posY'] for file in grid_files])
                max_posY = max([file['posY'] for file in grid_files])
                
                grids[i+1]['data'][file_key]['min_gridX'] = int(min_gridX)
                grids[i+1]['data'][file_key]['max_gridX'] = int(max_gridX)
                grids[i+1]['data'][file_key]['min_gridY'] = int(min_gridY)
                grids[i+1]['data'][file_key]['max_gridY'] = int(max_gridY)
                
                grids[i+1]['data'][file_key]['min_posX'] = int(min_posX)
                grids[i+1]['data'][file_key]['max_posX'] = int(max_posX)
                grids[i+1]['data'][file_key]['min_posY'] = int(min_posY)
                grids[i+1]['data'][file_key]['max_posY'] = int(max_posY)
                
                grids[i+1]['data'][file_key]['parentX'] = grid_files[0][pv[0]]
                grids[i+1]['data'][file_key]['parentY'] = grid_files[0][pv[1]]
                
            break
        
        return grids
                

    def grid_map(vp, layout):
        """
        Builds a grid map where file names are keys and the values are the
        level and variables.
        """
        
        grid_key = {}
        
        for level, pair in enumerate(layout[:1]):
            grids = [grid for grid in vp(group_by=pair)]
            
            if len(layout) - 1 == level:
                grid_vars = None
            
            else:
                grid_vars = layout[level+1]
            
            if grid_vars and len(grid_vars) > 1:
                width = max(vp.uniques[grid_vars[0]]) - min(vp.uniques[grid_vars[0]]) + 1
                height = max(vp.uniques[grid_vars[1]]) - min(vp.uniques[grid_vars[1]]) + 1
                grid_key[grid_vars] = {'width': width, 'height': height}
                
            if grid_vars and len(grid_vars) == 1:
                width = max(vp.uniques[grid_vars[0]]) - min(vp.uniques[grid_vars[0]]) + 1
                height = width
                grid_key[grid_vars] = {'width': width, 'height': height}
                
            for grid in grids:
                files = []
                x = []
                y = []
                gridX = []
                gridY = []
                
                for img in grid:
                    files.append(img['file'])
                    x.append(int(img['posX']))
                    y.append(int(img['posY']))
                    gridX.append(img['gridX'])
                    gridY.append(img['gridY'])
                    
                files.sort()
                
                grid_key[''.join(files)] = {
                    'position':(min(x), min(y)),
                    'width': max(x) - min(x) + 1,
                    'height': max(y) - min(y) + 1,
                    'pair': pair,
                    'level': level,
                    'gridX': (min(gridX), max(gridX)),
                    'gridY': (min(gridY), max(gridY)),
                    'variables': grid_vars
                    }
                
        return grid_key

    #TODO: Generate list of missing files for a given grid
    def infer_missing_files(files, vp):
        """
        Parses the input files to determine which variables are changing across
        the files. Generates all combinations of the changing files and returns
        which are missing in the input files.
        """
        
        pattern = vp.pattern
        fields = [f[1] for f in Formatter().parse(pattern) if f[1]]
        values = {field:[] for field in fields}
        
        # Get the values of each variable across the file/grid list
        for file in files:
            
            # Get the files values
            file_values = parse(pattern, file)
            
            for field in fields:
                
                # Add the files field values to the dictionary
                values[field].append(file_values[field])
                
        # Keep only the unique values (this removes duplicate values for
        # variables which do not vary across the grid)
        for f,v in values.items():
            
            if len(set(v)) > 1:
                # Convert the values to string and add leading zeros
                values[f] = [str(v).zfill(len(f)) for v in vp.uniques[f]]
           
            else:
                values[f] = [v[0]]
                
        all_files = []
        
        # Create all combinations of file names in grid
        for c in itertools.product(*[v for v in values.values()]):
            all_files.append(pattern.format(**{f:v for f,v in zip(fields, c)}))
        
        # Sort the list
        all_files.sort()
            
        return all_files


    def get_data_range(paths):
        
        data_range = {}
        df = pd.read_csv(paths[0])
        columns = df.columns
        data_range = {c:{'min':minv, 'max':maxv} for c,minv,maxv in zip(columns, df.min(), df.max())}
        
        for path in paths[1:]:
            df = pd.read_csv(path)
            df_min = df.min()
            df_max = df.max()
            columns = df.columns
            for c in columns:
                if df_min[c] < data_range[c]['min']:
                    data_range[c]['min'] = df_min[c]
                    
                if df_max[c] > data_range[c]['max']:
                    data_range[c]['max'] = df_max[c]

        return data_range
        

    def normalize(value, data_range, desired_range):
        
        m = value
        rmin = data_range['min']
        rmax = data_range['max']
        tmin = desired_range['min']
        tmax = desired_range['max']
        
        n = ((m-rmin)/(rmax-rmin))*(tmax-tmin) + tmin
        
        return n

   
    def generate_text(grids):
        
        text_layers = []
        
        for level_id, level in grids.items():
            
            text_cells = []
            pv = level['parent']
            width = level['width']
            height = level['height']
            
            for grid in level['data'].values():
            
                text = to_bijective(grid['parentY']) + str(grid['parentX'])
                position = (grid['min_posX'], grid['min_posY'])
                # print(text)
                # print(type(position))
                # print(type(position[0]))
                # print(type(position[1]))
            
                text_cells.append(
                    TextCell(
                        position=(grid['min_posX'], grid['min_posY']),
                        text=text
                        )
                    )
            
            # print(len(text_cells))
            # print(width)
            # print(height)
            assert len(text_cells) == width*height, 'Incorrect number of text cells'
            text_cells.sort()
            
            text_layers.append(
                TextLayerSpec(
                    id=level_id,
                    width=width,
                    height=height,
                    cell_size=3263,
                    # cell_width=cell_width,
                    # cell_height=cell_height,
                    data=text_cells
                    )
                )
            
        return text_layers
     
     
    def generate_heatmap(heatmap, layout, cell_size, vp, data_range, data_names=None):
        
        missing_files_out = Path('/home/ec2-user/polus-plugins/data/eastman/heatmap_data/missing_files.txt')
        missing_records = Path('/home/ec2-user/polus-plugins/data/eastman/heatmap_data/missing_records.txt')
        
        data = {'files':[], 'values':[]}
        grid_key = grid_map(vp, layout)
        data_cols = []
        values = []
        files = []
        meta = []
        if not data_names:
            data_names = []
            grid_cells = {}
        else:
            grid_cells = {name:[] for name in data_names}
        

        # keys = [k for k in grid_key.keys()]
        # pprint(keys[385])
        
        if heatmap:
            
            # TODO: Use itertools.groupby() to group the grids 
            with open(heatmap) as fr:
                
                for i, line in enumerate(fr):
                    columns = line.replace('\n', '').split(',')
                    
                    # If header find the filename column
                    if i == 0:
                        for j, c in enumerate(columns):
                            
                            if 'file' in c.lower() or 'image' in c.lower():
                                file_col = j
                            elif 'meta' in c.lower() or 'well' in c.lower():
                                meta_col = j
                            else:
                                data_cols.append(j)
                                data_names.append(c)
                                grid_cells[c] = []
                    
                    elif not values:
                        values = [float(columns[c]) for c in data_cols]
                        meta = columns[meta_col]
                        files.append(columns[file_col])
                        
                    else:
                        current = [float(columns[c]) for c in data_cols]
                            
                        if values == current and meta == columns[meta_col]:
                            files.append(columns[file_col])
                            
                        else:
                            files.sort()
                            data['files'].append(files)
                            data['values'].append(values)
                                
                            meta = columns[meta_col]
                            files = [columns[file_col]]
                            values = current
                            # grid_size = 9
                    
                # Add the last group
                data['files'].append(files)
                data['values'].append(values)
                
                fr.close()
                        
        else:
            # for key in grid_key.keys():
            #     # print(key)
            #     # if 'variables' not in grid_key[key]:
            #     #     print(grid_key[key])
            #     #     print('\n')
            #     #     print(key)
            #     if key != 'xy':
            #         print(grid_key[key])
            #         exit()
            data['files'] = [key for key in grid_key.keys() if key != 'xy' and 'c1.ome.tif' not in key]
            data['values'] = [[0]*len(data_names) for _ in range(len(data['files']))]
            
        widths = []
        heights = []
        cell_sizes = []
        i = 0
        # print(len(data['files']))
        
        for group, values in zip(data['files'], data['values']):
    
            group_key = ''.join(group)
            
            # if group_key not in grid_key:
                
            #     for i in range(len(group)):
            #         copy = group.copy()
            #         copy.pop(i)
            #         group_key = ''.join(group)
                    
            #         if group_key in grid_key:
            #             break
            
            # assert len(group) <= 9, 'Grid cell group must be less than or equal to 9'
            
            # If the grid cannot be found infer the missing files
            if group_key not in grid_key:
                # print(group_key)
                # print(grid_key)
                old_group = group.copy()
                group = infer_missing_files(group, vp)
                group_key = ''.join(group)
                assert len(group) <= 9, 'Grid cell group must be less than or equal to 9'
                
                
                for file in group:
                    if file not in old_group:
                        with open(missing_files_out, 'a') as fw:
                            fw.write(file+'\n')
            
                fw.close()

            position = grid_key[group_key]['position']
            position = (position[0] - 78712, position[1])
            
            if len(widths) < len(data_names):
                vars = grid_key[group_key]['variables']
                widths.append(grid_key[vars]['width'])
                heights.append(grid_key[vars]['height'])
            
            # print(data_names)
            for name, value in zip(data_names, values):
                # print(value)
                if heatmap:
                    nvalue = normalize(value, data_range[name], {'min':0, 'max':255})
                    assert 0 <= nvalue <= 255, 'Heatmap value outside of 0-255 range'
                    fill_color = (int(nvalue), 0, 0, 255)
                    
                else:
                    fill_color = (0, 0, 0, 1)
                grid_cells[name].append(
                    GridCell(
                        position=position,
                        fill_color=fill_color
                    )
                )
        
        # gls = {}
        gls = []
        
        if len(grid_cells['FPR-0.001']) < 384:
                with open(missing_records, 'a') as fw:
                    fw.write(heatmap.name + '\n')
                    fw.close()
        
        for name, width, height in zip(grid_cells.keys(), widths, heights):
            
            # Sort the grid cell by position
            grid_cells[name] = sorted(grid_cells[name])
            # print(len(grid_cells[name]))
            
            if len(grid_cells[name]) != 384:
                print(name)
                print(len(grid_cells[name]))
            
            assert len(grid_cells[name]) == 384, "Each grid cell spec must have 384 grid cells"
            
            g = GridCellLayerSpec(
                id = name, #+ '_' + ''.join(vp.output_name().split('.')[:-2]),
                width = width,
                height = height,
                # cell_height = cell_size,
                # cell_width = cell_size,
                cell_size = cell_size,
                data = grid_cells[name]
                )

            # if name in ['FPR-0.001','FPR-0.00001','OTSU']:
            # if name in ['FPR-0.001']:
            #     gls.append(g)
            
            gls.append(g)
                
            # print(gls)
            
        return gls, data_names


    def generate_chem_data(chem_dir, fill_holes=True):
        # Open and combine parquet smile data
        # chem = Path(__file__).parents[3].joinpath('data/eastman/smile_data/')
        # df = vaex.open(parquet_dir)
        # df = pd.read_parquet(chem_dir)

        # print(df.head())
        # print(df.tail())
        # print(df.describe())

        # Remove NA smile records
        #df = df.dropna(column_names=['smiles'])
        # df.dropna(subset=['smiles'], inplace=True)

        # Remove lower case letters from plate ID and store in new column
        # df.ai_assay_plate_id.str.replace('[a-z]', '')
        # pattern = re.compile('a-z')
        # df['ai_assay_plate_id'] = df.ai_assay_plate_id.apply(lambda x: pattern.sub('', x))

        # # Define columns to which are needed and which are not
        # to_keep = ['ai_assay_plate_id', 'smiles', 'ROW_INDEX', 'COL_INDEX']#, 'Client_Protocol_Viral_Replication']
        # to_drop = [c for c in df.columns if c not in to_keep]

        # # Drop the columns
        # # df.drop(to_drop, inplace=True)
        # df.drop(labels=to_drop, axis=1, inplace=True)

        # # Rename the plate id column
        # # df.rename('ai_assay_plate_id', 'plate_id')
        # df.rename({'ai_assay_plate_id': 'plate_id'}, axis=1, inplace=True)

        # # Convert col index to integer
        # df['COL_INDEX'] = df.COL_INDEX.astype(int)

        # df.drop_duplicates(subset=['ROW_INDEX', 'COL_INDEX', 'plate_id'], inplace=True, ignore_index=True)
        # df.rename({'ROW_INDEX':'row', 'COL_INDEX':'col', 'smiles':'smile'}, axis=1, inplace=True)

        # print(df.head())
        # print(df.tail())
        # print(df.describe())

        # unique_rows = df.unique('ROW_INDEX')
        # unique_cols = df.unique('COL_INDEX')
        # unique_plates = df.unique('plate_id')
        # unique_rows.sort()
        # unique_cols.sort()
        # unique_plates.sort()
        # print(unique_rows)
        # print(unique_cols)
        # print(len(unique_plates))
        dtype = {
            'plate':int, 
            'plate_id':str, 
            'virus_strain':str, 
            'smiles':str, 
            'compound_name':str, 
            'concentration':float
        }
        
        # Define and open the trace data
        trace_path = chem_dir.with_name('trace_control_final.csv')
        trace = pd.read_csv(trace_path,dtype=dtype)

        trace.drop_duplicates(['plate', 'well_name'], inplace=True, ignore_index=True)

        # # Define re pattern to extract the plate ID
        # pattern = re.compile('(?<=/)[V|0-9|_]*(?=-V|\[)')

        # # Define function to get the plate ID from directory
        # def parse_plate(direc, pattern):
        #     plate = pattern.search(direc).group(0).replace('_', '')
        #     return plate

        # # Parse the plate ID from the raw directory
        # trace['plate_id'] = trace.apply(lambda x: parse_plate(x.raw_directory, pattern), axis=1)

        trace.rename(
            {'plate_row':'row', 'plate_column':'col', 'virus_strain':'virus', 'compound_name':'compound', 'smiles':'smile'}, 
            axis=1, 
            inplace=True
            )
        
        # Define coluns to keep and which to drop
        to_keep = ['plate',  'row', 'col', 'virus', 'compound', 'smile', 'concentration']
        to_drop = [c for c in trace.columns if c not in to_keep]
        
        # Drop columns which are not needed
        trace.drop(to_drop, axis=1, inplace=True)
        
        # print(trace.shape)

        # print(trace.head())
        # print(trace.tail())
        # print(trace[~trace['compound'].isnull()])

        # Join on plate ID, this adds plate number i.e. 1,2,3,..,199,200 to data frame
        # df = df.merge(trace, on=['plate_id', 'row', 'col'], how='outer')
        
        # print(df.shape)
        # print(df[df['plate'].isnull()].shape)
        # print(df[df['plate'].isnull()])
        # print(df[~df['plate'].isnull()].shape)
        # print(df[~df['plate'].isnull()])
        # exit()
        
        # Drop the plate_id column
        # df.drop('plate_id', axis=1, inplace=True)

        # print(df.head())
        # print(df.tail())
        # print(df[~df['compound'].isnull()])
        # print(df.describe())
        
        trace['compound'] = trace['compound'].apply(lambda x: '' if x=='None' else x)
        trace['smile'] = trace['smile'].apply(lambda x: '' if x=='None' else x)
        
        # trace['compound'] = trace['compound'].apply(lambda x: pd.NA if x=='None' else x)
        # trace['smile'] = trace['smile'].apply(lambda x: pd.NA if x=='None' else x)
        
        # trace.fillna('', inplace=True)
        # print(trace.head())
        # print(trace.tail())
        # print(trace.describe())
        # print(trace[~trace['compound'].isnull()].shape)
        # print(trace[~trace['virus'].isnull()].shape)
        # print(trace[~trace['smile'].isnull()].shape)
        # print(trace[trace['compound'] == 'None'])
        # print(trace[trace['smile'] == 'None'])
        # print(trace[trace['virus'] == 'None'])
        
        
        # if fill_holes:
        #     p = [i for i in range(1, 201)]
        #     y = [i for i in range(1, 17)]
        #     x = [i for i in range(1, 25)]

        #     all_wells = [c for c in itertools.product(p,y,x)]
        #     full = pd.DataFrame(all_wells, columns=['plate', 'row', 'col'])
        #     df = full.merge(df, how='outer', on = ['plate', 'row', 'col'])
        #     df.fillna('', inplace=True)
        
        trace.fillna('', inplace=True)
        # trace.dropna(subset='smile', inplace=True)

        return trace.groupby('plate')


    def generate_chem(df, plate_id):

        def generate_chem_cell(x, y, smile, virus, compound, concentration):
            
            posX = (1081*3+20)*(x-1)
            posY = (1081*3+20)*(y-1)
            position = (posX, posY)

            chem_cell = ChemCell(
                position = position,
                smile = smile
                )
            
            # chem_cell = TextCell(
            #     position = position,
            #     text = smile
            #     )
            
            viral_cell = TextCell(
                position = position,
                text = virus
            )
            
            assert isinstance(concentration, float) or concentration == '', \
                'Concentration must be a float or empty string'
            
            if isinstance(concentration, float):
                compound_text = '{} ({:,} nM)'.format(compound, concentration)
            else:
                compound_text = compound
            
            compound_cell = TextCell(
                position = position,
                text = compound_text
            )
            
            return chem_cell, viral_cell, compound_cell
        

        cells = [
            cell for cell in df.apply(
                lambda x: generate_chem_cell(
                    x=x.col, y=x.row, smile=x.smile, virus=x.virus, compound=x.compound, concentration=x.concentration
                    ), 
                axis=1
                )
            ]
        
        chem_cells, viral_cells, compound_cells = map(list, zip(*cells))
        
        chem_cells = sorted(chem_cells)
        viral_cells = sorted(viral_cells)
        compound_cells = sorted(compound_cells)
        
        cls = ChemLayerSpec(
            id=plate_id,
            width=24,
            height=16,
            # cell_width=3263,
            # cell_height=3263,
            cell_size=3263,
            data=chem_cells
        )

        # cls = TextLayerSpec(
        #     id=plate_id,
        #     width=24,
        #     height=16,
        #     # cell_width=3263,
        #     # cell_height=3263,
        #     cell_size=3263,
        #     data=chem_cells
        # )
        
        vls = TextLayerSpec(
            id='virus', #+ plate_id,
            width=24,
            height=16,
            # cell_width=3263,
            # cell_height=3263,
            cell_size=3263,
            data=viral_cells
        )
        
        mls = TextLayerSpec(
            id='compound', #+ plate_id,
            width=24,
            height=16,
            # cell_width=3263,
            # cell_height=3263,
            cell_size=3263,
            data=compound_cells
        )
        
        
        return [cls], vls, mls
    
    
    logger.info('Parsing stitching vector paths...')
    stitching_vector_paths = []
    # Define the stitching vector path
    stitching_vector_paths = [path for path in stitching_vector.rglob('img-global-positions*')]
    stitching_vector_paths.sort()
    # stitching_vector_paths = stitching_vector_paths[:4]
    # for path in Path(stitching_vector).iterdir():
    #     if 'img-global-positions' in path.name:
    #         stitching_vector_paths.append(path)
    
    # If not passed infer the filepattern
    if file_pattern is None:
        logger.info("Infering filepattern...")
        
        files = []
        
        # Parse the image file names in the stiching vector
        with stitching_vector_paths[0].open('r') as fhand:
            for line in fhand.readlines():
                for component in line.split("; "):
                    elements = component.split(": ")
                    if elements[0] == "file":
                        file = elements[1]
                        files.append(file)
                        break
        
        vector_pattern = filepattern.infer_pattern(files)
        logger.info("Inferred filepattern = {}".format(vector_pattern))
        
        
    if heatmap:
        heatmap_paths = [path for path in heatmap.iterdir()]
        heatmap_paths.sort()
        # heatmap_paths = heatmap_paths[35:37]
        data_range = get_data_range(heatmap_paths)
        
    else:
        heatmap_paths = [None]*len(stitching_vector_paths)
        
    
    if chem:
        chem_groupy_plate = generate_chem_data(chem)
        
    if concatenate:
        # overlay_outputs = {}
        overlay_outputs = []
        
    logger.info('Generating overlays...')
    for vector_path in tqdm(stitching_vector_paths):
        vp = filepattern.VectorPattern(
            file_path=vector_path, 
            pattern=file_pattern
            )
        
        # print(vp.uniques)
        
        # Get the assembled image layout and variable details
        var_stats, layout = get_var_stats(vp, img_size=(1080,1080))
        grid = [f[0] for f in vp()]
        layout = layout.split(',')
        
        grids = build_grids(vp, layout)
        # pprint(grids)
        
        # pprint(var_stats)
        
        # Generate an ouput name and path for the image's overlays
        output_name = ''.join(vp.output_name().split('.')[:-2])
        output_path = outDir.joinpath(output_name).with_suffix('.json')

        # Generate the text overlays
        if text:
            # text_layers = generate_text(layout, grid, var_stats, vp)
            text_layers = generate_text(grids)
            
        else:
            text_layers = None
        
        if heatmap:
            
            # Get the heatmap path which matches the vector path
            plate = vector_path.parent.name.split('_')[0].replace('p', 'plate')
            heatmap_paths = [path for path in heatmap.iterdir() if plate in path.name]
            assert len(heatmap_paths) <=  1, 'Cannot match heatmap path with vector path'
            
            if heatmap_paths:
                heatmap_path = heatmap_paths[0]
                # print(heatmap_path)
                grid_cell_layers, data_names = generate_heatmap(heatmap_path, layout, 3263, vp, data_range)
            
            else:
                grid_cell_layers, data_names = generate_heatmap(None, layout, 3263, vp, data_names, data_range)
            
        else:
            grid_cell_layers = None
            
        
        if chem:
            
            # Get the current vector pattern plate number
            p = grid[0]['p']
            
            # Get the current plate group and generate cls if plate group exists
            if p in chem_groupy_plate.groups:
                plate_group = chem_groupy_plate.get_group(p)
                chem_layers, viral_layers, compound_layers = generate_chem(plate_group, output_name)
                
                text_layers.extend([viral_layers, compound_layers])
                # text_layers.extend([compound_layers])
                
            else:
                chem_layers = None
                viral_layers = None
                compound_layers = None
            
        else:
            chem_layers = None
            viral_layers = None
            compound_layers = None
            
        
        ols = OverlaySpec(
            grid_cell_layers=grid_cell_layers,
            text_layers=text_layers, 
            chem_layers=chem_layers
            )
        
        # ols = OverlaySpec(
        #     text_layers=chem_layers
        # )
        
        # text_overlay = OverlaySpec(
        #     text_layers=text_layers,
        # )
        
        with output_path.open("w") as fw:
            json.dump(
                ols.dict(by_alias=True),
                fw,
                indent=2,
            )

        # text_output = output_path.with_name('text_' + output_path.name)
        
        # heatmap_overlays = {name: OverlaySpec(grid_cell_layers=[gls]) for name, gls in grid_cell_layers.items()}
        # heatmap_outputs = {name: output_path.with_name(name + output_path.name) for name in heatmap_overlays.keys()}
        
        # with text_output.open("w") as fw:
        #     json.dump(
        #         text_overlay.dict(by_alias=True),
        #         fw,
        #         indent=2,
        #     )
        
        # for name in heatmap_overlays.keys():
            
        #     output = heatmap_outputs[name]
        #     ols = heatmap_overlays[name]
            
        #     with output.open("w") as fw:
        #         json.dump(
        #             ols.dict(by_alias=True),
        #             fw,
        #             indent=2,
        #         )         
            
        if concatenate:
            
            # overlay_outputs['text'] = overlay_outputs.get('text', []) + [text_output]
            
            # for name, output in heatmap_outputs.items():
            #     overlay_outputs[name] = overlay_outputs.get(name, []) + [output]
            overlay_outputs.append(output_path)
        
    if concatenate:
        
        def combine_overlays(paths, output):
            
            num_paths = len(paths)
            logger.info('Combining {} overlay files in {}'.format(num_paths, output))
            
            with open(output, 'w') as fw:
                fw.write('[')
                fw.close()
                
            # Combine all the overlay files
            for i, path in enumerate(paths):
                
                # Load the overlay data
                with path.open('r') as fr:
                    data = json.load(fr)
                    fr.close()
            
                # Write the overlay data
                with output.open('a') as fa:
                    
                    fa.write(json.dumps(data, indent=2))
                    
                    if i != num_paths - 1:
                        fa.write(',\n')
                        
                    else:
                        fa.write(']')
                        
                fa.close()

        
        # for name, paths in overlay_outputs.items():
            
        #     paths.sort()
        #     output = outDir.joinpath(name + '_overlay.json')
        #     combine_overlays(paths, output)

        combined_output = outDir.joinpath('overlay.json')
        
        combine_overlays(overlay_outputs, combined_output)
             


if __name__ == '__main__':

    # Setup Command Line Arguments
    logger.info("Parsing arguments...")

    # Instantiate argparser object
    parser = argparse.ArgumentParser(
        prog="main", description="Polus Render Overlay Plugin"
    )

    # Add the plugin arguments
    parser.add_argument(
        "--stitchingVector", 
        dest="stitchingVector", 
        type=str, 
        help="Path to the stitching vector directory", 
        required=True
    )

    parser.add_argument(
        "--outDir", 
        dest="outDir", 
        type=str, 
        help="Path to the stitching vector directory", 
        required=True
    )

    parser.add_argument(
        "--filePattern", 
        dest="filePattern", 
        type=str, 
        help="The stitching vector file pattern", 
        required=False
        )
    
    parser.add_argument(
        "--heatmap",
        dest="heatmap",
        type=str,
        help="Directory of the heatmap specifications", 
        required=False,
    )
    
    parser.add_argument(
        "--chem",
        dest="chem",
        type=str,
        help="Directory to the chem (dmile) specifications", 
        required=False,
    )    

    parser.add_argument(
        "--text", 
        dest="text",
        action=argparse.BooleanOptionalAction,
        help="If text overlays should generated", 
        required=False,
        )

    parser.add_argument(
        "--concatenate", 
        dest="concatenate",
        action=argparse.BooleanOptionalAction,
        help="If the output be a single file", 
        required=False,
        )


    # Parse and log the arguments
    args = parser.parse_args()
    
    stitching_vector = Path(args.stitchingVector)
    logger.info('stitchingVector = {}'.format(stitching_vector))

    outDir = Path(args.outDir)
    logger.info('outDir = {}'.format(outDir))

    file_pattern = args.filePattern
    logger.info('filePattern = {}'.format(file_pattern))

    if args.heatmap:
        heatmap = Path(args.heatmap)
    else:
        heatmap = args.heatmap
    logger.info('heatmap = {}'.format(heatmap))

    if args.chem:
        chem = Path(args.chem)
    else:
        chem = args.chem
    logger.info('chem = {}'.format(chem))
    
    if args.text:
        text = args.text
    else:
        text = False
    logger.info('text = {}'.format(text))
    
    if args.concatenate:
        concatenate = args.concatenate
    else:
        concatenate = False
    logger.info('concatenate = {}'.format(concatenate))

    main(
        stitching_vector=stitching_vector,
        outDir=outDir,
        file_pattern=file_pattern,
        heatmap=heatmap,
        chem=chem,
        text=text,
        concatenate=concatenate
        )
