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
    def generate_text(layout, grid, var_stats, vp) -> dict:
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
            # if level != 1:
            #     continue
            
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
                cell_width=cell_width,
                cell_height=cell_height,
                data=overlay
                )

            text_layers.append(tls)
        
        return text_layers
    
    
    def build_grids(vp, sub_grids={}):
        
        grids = {}
        
        if not sub_grids:
            sub_grids = [file[0] for file in vp()]
            
        variables = vp.variables
            
        for g in sub_grids[0:1]:
            for v in variables:
                print('{} = {}'.format(v, g[v]))
                

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
   
     
    def generate_heatmap(heatmap, layout, cell_size, vp):
        
        missing_files_out = heatmap.parents[1].joinpath('missing_files.txt')
        
        data = {'files':[], 'values':[]}
        grid_key = grid_map(vp, layout)
        data_cols = []
        data_names = []
        values = []
        files = []
        meta = []
        grid_cells = {}
        gls = []
        
        # keys = [k for k in grid_key.keys()]
        # pprint(keys[385])
        
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
                        grid_size = 9
                        
        
        fr.close()
        widths = []
        heights = []
        cell_sizes = []
        i = 0
        for group, values in zip(data['files'], data['values']):
    
            group_key = ''.join(group)
            
            # if group_key not in grid_key:
                
            #     for i in range(len(group)):
            #         copy = group.copy()
            #         copy.pop(i)
            #         group_key = ''.join(group)
                    
            #         if group_key in grid_key:
            #             break
            
            # If the grid cannot be found infer the missing files
            if group_key not in grid_key:
                old_group = group.copy()
                group = infer_missing_files(group, vp)
                group_key = ''.join(group)
            
                for file in group:
                    if file not in old_group:
                        with open(missing_files_out, 'a') as fw:
                            fw.write(file+'\n')
            
                fw.close()
                
            position = grid_key[group_key]['position']
            
            if len(widths) < len(data_names):
                vars = grid_key[group_key]['variables']
                widths.append(grid_key[vars]['width'])
                heights.append(grid_key[vars]['height'])
            
            for name, value in zip(data_names, values):
                grid_cells[name].append(
                    GridCell(
                        position=position,
                        fill_color=(value, 0, 0, 255)
                    )
                )
        
        for name, width, height in zip(grid_cells.keys(), widths, heights):
            
            # Sort the grid cell by position
            grid_cells[name] = sorted(grid_cells[name])
            
            gls.append(
                GridCellLayerSpec(
                    id = name,
                    width = width,
                    height = height,
                    cell_height = cell_size,
                    cell_width = cell_size,
                    data = grid_cells[name]
                )
            )
            
        return gls


    def generate_chem_data(chem_dir, fill_holes=True):
        # Open and combine parquet smile data
        # chem = Path(__file__).parents[3].joinpath('data/eastman/smile_data/')
        # df = vaex.open(parquet_dir)
        df = pd.read_parquet(chem_dir)

        # print(df.head())
        # print(df.tail())
        # print(df.describe())

        # Remove NA smile records
        #df = df.dropna(column_names=['smiles'])
        df.dropna(subset=['smiles'], inplace=True)

        # Remove lower case letters from plate ID and store in new column
        # df.ai_assay_plate_id.str.replace('[a-z]', '')
        pattern = re.compile('a-z')
        df['ai_assay_plate_id'] = df.ai_assay_plate_id.apply(lambda x: pattern.sub('', x))

        # Define columns to which are needed and which are not
        to_keep = ['ai_assay_plate_id', 'smiles', 'ROW_INDEX', 'COL_INDEX']#, 'Client_Protocol_Viral_Replication']
        to_drop = [c for c in df.columns if c not in to_keep]

        # Drop the columns
        # df.drop(to_drop, inplace=True)
        df.drop(labels=to_drop, axis=1, inplace=True)

        # Rename the plate id column
        # df.rename('ai_assay_plate_id', 'plate_id')
        df.rename({'ai_assay_plate_id': 'plate_id'}, axis=1, inplace=True)

        # Convert col index to integer
        df['COL_INDEX'] = df.COL_INDEX.astype(int)

        df.drop_duplicates(subset=['ROW_INDEX', 'COL_INDEX', 'plate_id'], inplace=True, ignore_index=True)

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

        # Define and open the trace data
        trace_path = chem_dir.with_name('trace_control.csv')
        trace = pd.read_csv(trace_path)

        trace.drop_duplicates('raw_directory', inplace=True, ignore_index=True)

        # Define re pattern to extract the plate ID
        pattern = re.compile('(?<=/)[V|0-9|_]*(?=-V|\[)')

        # Define function to get the plate ID from directory
        def parse_plate(direc, pattern):
            plate = pattern.search(direc).group(0).replace('_', '')
            return plate

        # Parse the plate ID from the raw directory
        trace['plate_id'] = trace.apply(lambda x: parse_plate(x.raw_directory, pattern), axis=1)

        # Define coluns to keep and which to drop
        to_keep = ['plate', 'plate_id']
        to_drop = [c for c in trace.columns if c not in to_keep]

        # Drop columns which are not needed
        trace.drop(to_drop, axis=1, inplace=True)

        # Join on plate ID, this adds plate number i.e. 1,2,3,..,199,200 to data frame
        df = df.merge(trace, on='plate_id', how='inner')
        
        # Drop the plate_id column
        df.drop('plate_id', axis=1, inplace=True)

        df.rename({'ROW_INDEX':'row', 'COL_INDEX':'col', 'smiles':'smile'}, axis=1, inplace=True)

        # print(df.head())
        # print(df.tail())
        # print(df.describe())
        
        if fill_holes:
            p = [i for i in range(1, 201)]
            y = [i for i in range(1, 17)]
            x = [i for i in range(1, 25)]

            all_wells = [c for c in itertools.product(p,y,x)]
            full = pd.DataFrame(all_wells, columns=['plate', 'row', 'col'])
            df = full.merge(df, how='outer', on = ['plate', 'row', 'col'])
            df.fillna('', inplace=True)


        return df.groupby('plate')


    def generate_chem(df, plate_id):

        def generate_chem_cell(x, y, smile):
            
            posX = (1081*3+20)*(x-1)
            posY = (1081*3+20)*(y-1)
            position = (posX, posY)

            chem_cell = ChemCell(
                position = position,
                smile = smile
                )
            
            return chem_cell

        chem_cells = [
            cell for cell in df.apply(
                lambda x: generate_chem_cell(x=x.col, y=x.row, smile=x.smile), 
                axis=1
                )
            ]
        
        chem_cells = sorted(chem_cells)
        
        cls = ChemLayerSpec(
            id=plate_id,
            width=24,
            height=16,
            cell_width=3263,
            cell_height=3263,
            data=chem_cells
        )
        
        return [cls]
    
    logger.info('Parsing stitching vector paths...')
    stitching_vector_paths = []
    # Define the stitching vector path
    stitching_vector_paths = [path for path in stitching_vector.rglob('img-global-positions*')]
    stitching_vector_paths.sort()
    stitching_vector_paths = stitching_vector_paths
    # stitching_vector_paths = stitching_vector_paths[190:]
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

    
    if concatenate:
        all_overlay_paths = []
        
    if heatmap:
        heatmap_paths = [path for path in heatmap.iterdir()]
        heatmap_paths.sort()
        # heatmap_paths = heatmap_paths[:10]
        
    else:
        heatmap_paths = [None]*len(stitching_vector_paths)
        
    
    if chem:
        chem_groupy_plate = generate_chem_data(chem)
        
        
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
        
        # pprint(var_stats)
        
        # Generate an ouput name and path for the image's overlays
        output_name = ''.join(vp.output_name().split('.')[:-2])
        output_path = outDir.joinpath(output_name).with_suffix('.json')

        # Generate the text overlays
        if text:
            text_layers = generate_text(layout, grid, var_stats, vp)
            
        else:
            text_layers = None
        
        if heatmap:
            
            # Get the heatmap path which matches the vector path
            plate = vector_path.parent.name.split('_')[0].replace('p', 'plate')
            heatmap_paths = [path for path in heatmap.iterdir() if plate in path.name]
            assert len(heatmap_paths) <=  1, 'Cannot match heatmap path with vector path'
            
            if heatmap_paths:
                heatmap_path = heatmap_paths[0]
                grid_cell_layers = generate_heatmap(heatmap_path, layout, 3263, vp)
            
            else:
                grid_cell_layers = None
            
        else:
            grid_cell_layers = None
            
        
        if chem:
            
            # Get the current vector pattern plate number
            p = grid[0]['p']
            
            # Get the current plate group and generate cls if plate group exists
            if p in chem_groupy_plate.groups:
                plate_group = chem_groupy_plate.get_group(p)
                chem_layers = generate_chem(plate_group, output_name)
                
            else:
                chem_layers = None
            
        else:
            chem_layers = None
            
        
        overlay = OverlaySpec(
            grid_cell_layers=grid_cell_layers,
            text_layers=text_layers, 
            chem_layers=chem_layers
            
            )
        
        # print(output_path)
        
        with output_path.open("w") as fw:
            json.dump(
                overlay.dict(by_alias=True),
                fw,
                indent=2,
            )
            
        if concatenate:
            all_overlay_paths.append(output_path)
        
    if concatenate:
        
        all_overlay_paths.sort()
        combined_output_path = outDir.joinpath('overlay.json')
        num_paths = len(all_overlay_paths)
        logger.info('Combining {} overlay files in {}'.format(num_paths, combined_output_path))
        
        # Create the overlay fine with single open array bracket
        with open(combined_output_path, 'w') as fw:
            fw.write('[')
            fw.close()
        
        # Combine all the overlay files
        for i, path in enumerate(all_overlay_paths):
            
            # Load the overlay data
            with path.open('r') as fr:
                data = json.load(fr)
                fr.close()
        
            # Write the overlay data
            with combined_output_path.open('a') as fa:
                
                fa.write(json.dumps(data, indent=2))
                
                if i != num_paths - 1:
                    fa.write(',\n')
                    
                else:
                    fa.write(']')
                    
                fa.close()
                
    # build_grids(vp)


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
