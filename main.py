import cv2
import numpy as np
from fastsam import FastSAM, FastSAMPrompt
import torch
from PIL import Image
from nicegui import ui, events, app
import base64
import io
import tempfile
import statistics
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde


uploaded_image_content = None

@ui.page('/')
def homePage():
    with ui.header(elevated=True).style('background-color: white'):
        ui.image('./IMGS/SmartMCsLogo.png').style('width: 45px')
    with ui.column().classes('w-full items-center'):
        ui.label('Particle Image Analysis').style('font-size: 280%; margin: 25px;')
        ui.label('Select a model to analyse your image. Both models will return a particle count and a list of diameters(um). ').style('font-size: 125%;')

        with ui.row().classes('w-full').style('justify-content: center; margin: 20px'):
            with ui.column().style('width: 25%; margin-right: 15px;'):
                with ui.card().classes('w-full items-center').style('padding: 0px;'):
                    ui.image('./IMGS/standard.png').style('height: 120px')
                    ui.label('Circular Particle Analysis').style('font-size: 175%;')
                    ui.button('START', on_click=lambda: ui.navigate.to('/circularCount')).classes('w-full').style('font-size: 125%')
                ui.label('This model uses HoughCircles from OpenCV to detect circular edges in the image.').style('font-size: 125%; margin: 10px 0px;')
                ui.label('Features:').style('font-size: 125%')
                with ui.list().props('dense separator').style('font-size: 125%'):
                    ui.item('- Detects circular MCs only')
                    ui.item('- Fast processing')
                    ui.item('- Can detect images with high overlap')

            with ui.column().style('width: 25%; margin-left: 15px;'):
                with ui.card().classes('w-full items-center').style('padding: 0px;'):
                    ui.image('./IMGS/SAMpic.png').style('height: 120px')
                    ui.label('Irregular Particle Analysis').style('font-size: 175%')
                    ui.button('START', on_click=lambda: ui.navigate.to('/irregularCount')).classes('w-full').style('font-size: 125%')
                ui.label('This model builds off the segment-anything model to generate masks on an image.').style('font-size: 125%; margin: 10px 0px;')
                ui.label('Features:').style('font-size: 125%')
                with ui.list().props('dense separator').style('font-size: 125%'):
                    ui.item('- Can detect irregular MCs')
                    ui.item('- Slower processing')
                    ui.item('- Can only detect images with low detail and concentration')

@ui.page('/calculate')
def calculate():
    with ui.header(elevated=True).style('background-color: white'):
        ui.image('./IMGS/SmartMCsLogo.png').style('width: 45px')        
    stored_data = app.storage.user.get('data', {})

    if len(stored_data) == 0:
        ui.notify("No diameter data available.", type="warning")
        return
    ui.space()

    # Initialize variables
    dil = 4
    drop = 5
    mass = 0.1
    massIn = 10000
    beads_needed = 11

    # Function to update tables
    def update_tables():
        dias = stored_data.get('dias', [])
        length = len(dias)
        average_dia = statistics.mean(dias) if dias else 0
        avg_radius_um = average_dia / 2
        avg_radius_cm = avg_radius_um / 10000
        avg_SA = 4 * math.pi * avg_radius_cm ** 2
        beads_needed_per_SA = beads_needed / avg_SA
        undilute = length * dil
        countPeruL = undilute / drop
        totalNumPermL = countPeruL * massIn / 2
        totalSApermL = avg_SA * totalNumPermL
        totalBeadsPerGram = totalNumPermL / mass

        # Update Result Values Table
        diaList.rows = [
            {'key': 'Count', 'value': f'{length}'},
            {'key': 'Avg. Diameter (um)', 'value': f'{average_dia:.8f}'},
            {'key': 'Avg. Radius (um)', 'value': f'{avg_radius_um:.8f}'},
            {'key': 'Avg. Radius (cm)', 'value': f'{avg_radius_cm:.8f}'},
            {'key': 'Avg. SA (cm^2)', 'value': f'{avg_SA:.8f}'},
        ]

        # Update Further Calculations Table
        tableCalc.rows = [
            {'key': 'Undiluted Count', 'value': f'{(undilute):.5f}'},
            {'key': 'Count per uL (beads/uL)', 'value': f'{(countPeruL):.5f}'},
            {'key': 'Total no./mL (beads/mL)', 'value': f'{totalNumPermL:.5f}'},
            {'key': 'Total SA/mL (cm^2/mL)', 'value': f'{totalSApermL:.5f}'},
            {'key': 'Total SA/g (cm^2/g)', 'value': f'{(totalSApermL / mass):.5f}'},
            {'key': 'Total no. of beads per gram (beads/g)', 'value': f'{(totalBeadsPerGram):.5f}'},
            {'key': 'Mass of carrier needed per mL of culture (g/mL)', 'value': f'{(beads_needed_per_SA / totalBeadsPerGram):.5f}'}
        ]
    
        # Function to plot histogram and density curve
    def plot_histogram_and_density(dias):
        if not dias:
            ui.notify("No diameter data to plot.", type="warning")
            return

        # Create a figure for the plot
        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot histogram
        ax.hist(dias, bins=10, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Diameter')

        kde = gaussian_kde(dias)
        x_min = min(dias) - 0.4 * (max(dias) - min(dias)) 
        x_max = max(dias) + 0.4 * (max(dias) - min(dias))  
        x_range = np.linspace(x_min, x_max, 1000)
        kde_values = kde(x_range)

        mean_dia = np.mean(dias)
        std_dia = np.std(dias)

        ax.plot(x_range, kde_values, color='blue', label=f'Density Curve\nMean: {mean_dia:.2f}, Std: {std_dia:.2f}')

        ax.set_xlabel('Diameter (um)')
        ax.set_ylabel('Density')
        ax.set_title('Diameter distribution of particles')
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        plot_container.clear() 
        plot_container.set_source(f'data:image/png;base64,{img_base64}')

    with ui.row().classes('w-full items-center').style('margin-bottom: 20px;'):
        ui.button('Back', on_click=ui.navigate.back).style('margin-left: 10px')
        ui.label('Particle Data Analysis').style('font-size: 280%; text-align: center; display: block; margin-left: auto; margin-right: auto;')

    with ui.row().classes('w-full no-wrap flex-start').style('justify-content: space-evenly;'):

        # Left Pane: Details
        with ui.card().classes('items-center').style('padding: 3%'):
            with ui.row().classes('items-center'):
                ui.label('Change Input Values').style('font-size: 150%')
            with ui.row().classes('items-center').style('width: 100%'):
                with ui.column().classes('items-center').style('width: 100%'):
                    dil_input = ui.input('Enter dilution number', value=4).style('width: 80%')
                    drop_input = ui.input('Enter droplet volume (mL)', value=5).style('width: 80%')
                    mass_input = ui.input('Mass (g)', value=0.1).style('width: 80%')
                    mass_in_input = ui.input('Mass in (uL)', value=10000).style('width: 80%')
                    beads_needed_input = ui.input('Beads needed per SA', value=11).style('width: 80%')
                    ui.space()

                    def calculate_values():
                        # Update the variables with the input values
                        nonlocal dil, drop, mass, massIn, beads_needed
                        dil = int(dil_input.value)
                        drop = float(drop_input.value)
                        mass = float(mass_input.value)
                        massIn = float(mass_in_input.value)
                        beads_needed = float(beads_needed_input.value)
                        update_tables()

                    ui.button('Submit', on_click=calculate_values)

        # Right Pane: Image Display
        with ui.card().classes('items-center').style('padding: 3%; width: 60%'):
            dias = stored_data.get('dias', [])

            async def copy_new_results():
                rows = diaList.rows
                clipboard_data = "key\tvalue\n"
                ui.notify("Table successfully copied to clipboard.", type="positive")
                for row in rows:
                    clipboard_data += f"{row['key']}\t{row['value']}\n"
                await ui.run_javascript(f'navigator.clipboard.writeText(`{clipboard_data}`)')

            async def copy_calc():
                rows = tableCalc.rows
                clipboard_data = "key\tvalue\n"
                ui.notify("Table successfully copied to clipboard.", type="positive")
                for row in rows:
                    clipboard_data += f"{row['key']}\t{row['value']}\n"
                await ui.run_javascript(f'navigator.clipboard.writeText(`{clipboard_data}`)')

            # Initial calculation and table setup
            with ui.row().classes('items-center'):
                ui.label('Further Analysis Results').style('font-size: 150%;')
            with ui.row():
                with ui.column().classes('items-center'):
                    diaList = ui.table(
                        columns=[
                            {'name': 'key', 'label': 'Key', 'field': 'key', 'align': 'left'},
                            {'name': 'value', 'label': 'Value', 'field': 'value', 'align': 'right'},
                        ],
                        rows=[],
                        row_key='key', 
                        title='Result Values'
                    ).style('margin: 10px')
                    ui.button('Copy Result Values', on_click=copy_new_results)
                with ui.column().classes('items-center'):
                    tableCalc = ui.table(
                        columns=[
                            {'name': 'key', 'label': 'Key', 'field': 'key', 'align': 'left'},
                            {'name': 'value', 'label': 'Value', 'field': 'value', 'align': 'right'},
                        ],
                        rows=[],
                        row_key='key', 
                        title='Further Calculations'
                    ).style('margin: 10px')
                    ui.button('Copy Calculations', on_click=copy_calc)    
            plot_container = ui.image().style('width: 900px; margin-top: 20px')  # Image placeholder for plot


            # Update tables with initial values
            update_tables()
            plot_histogram_and_density(dias)
            

@ui.page('/circularCount')
def circularMC():
    global uploaded_image_content
    uploaded_image_content = None

    with ui.header(elevated=True).style('background-color: white'):
        ui.image('./IMGS/SmartMCsLogo.png').style('width: 45px')        
    
    async def on_submit():
        global uploaded_image_content
        
        if uploaded_image_content is None:
            ui.notify("Please upload an image before submitting.", type="warning")
            return

        # Clear the previous image and results
        image_container.clear()
        result_container.clear()
        

        factor = float(mag.value)
        if factor <= 0: 
            ui.notify("Error: Invalid scale factor", type="warning")
            return

        # Convert the uploaded file to an OpenCV image
        nparr = np.frombuffer(uploaded_image_content, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if original_image is None:
            ui.notify("Error: Could not read the image.")
            return

        min_radius = int(min_radius_input.value) * factor
        max_radius = int(max_radius_input.value) * factor
        dist_level = int(int(dist_input.value) * factor)
        sens_level = 1 if sens_input.value == 100 else 100 - int(sens_input.value)


        total_count, total_diameters, output_image = process_circular_image(original_image, int(min_radius), int(max_radius), dist_level, sens_level)
        
        # Convert the processed image to base64 and update the UI 
        _, buffer = cv2.imencode('.jpg', output_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        image_container.set_source(f'data:image/jpeg;base64,{base64_image}')


        validDias = []

        for dia in total_diameters:
            if (dia >= (min_radius * 2)) and (dia <= (max_radius * 2)):
                validDias.append(dia / factor)
        validDias.sort()

        def open_calculate():
            new = {'dias': validDias} 
            app.storage.user.update(data=new) 
            ui.navigate.to('calculate')

        def open_graph():
            new = {'dias': total_diameters} 
            app.storage.user.update(data=new) 
            ui.navigate.to('graph')


        # Create the analysis results card dynamically
        with result_container:
            with ui.card().classes('items-center').style('width: 100%'):
                ui.label('Analysis Results').style('font-size: 125%')
                with ui.row().style('width: 100%'):
                    with ui.column().classes('items-center').style('width: 60%'):
                        with ui.row().style('width: 100%').classes('items-center'):
                            with ui.column().classes('items-center').style('width: 45%'):
                                ui.label('Particle Count: ')
                                ui.label(total_count).style('font-size: 300%')
                                ui.space()

                            with ui.column().classes('items-center').style('width: 45%'):
                                ui.label('Avg. Diameter (um): ')
                                mean = 0 if len(validDias) == 0 else statistics.mean(validDias)
                                ui.label(f"{mean:.3f}").style('font-size: 300%')
                                ui.space()

                            with ui.column().classes('items-center').style('width: 45%'):
                                ui.label('Stdev (um): ')
                                std = 0 if len(validDias) == 0 else (statistics.pstdev(validDias))
                                ui.label(f"{std:.3f}").style('font-size: 300%')
                                ui.space()

                            with ui.column().classes('items-center').style('width: 45%'):
                                ui.label('Coeff. of variation: ')
                                coeff = 0 if len(validDias) == 0 else (statistics.pstdev(validDias)/mean)
                                ui.label(f"{coeff:.3f}").style('font-size: 300%')
                                ui.space()
                        with ui.row().style('width: 100%, margin: 10px').classes('items-center'):
                            ui.label('Click here for further calculations and a graph representation.')
                        ui.button('calculate',  on_click=open_calculate)
  

                    with ui.column().classes('items-center').style('width: 30%'):
                        ui.label('List of Diameters(um): ')
                        async def copy_table_to_clipboard():
                            rows = diaList.rows
                            clipboard_data = "MC Index\tdia(um)\n"
                            ui.notify("Table successfully copied to clipboard.", type="positive")
                            for row in rows:
                                clipboard_data += f"{row['MC Index']}\t{row['dia(um)']}\n"
                            await ui.run_javascript(f'navigator.clipboard.writeText(`{clipboard_data}`)')

                        ui.button('Copy to Clipboard', on_click=copy_table_to_clipboard)
                        diaList = ui.table(
                            columns=[
                                {'name': 'MC Index', 'label': 'MC Index', 'field': 'MC Index', 'align': 'left'},
                                {'name': 'dia(um)', 'label': 'Diameter (um)', 'field': 'dia(um)', 'align': 'right'},
                            ],
                            rows=[{'MC Index': j + 1, 'dia(um)': f'{dia:.3f}'} for j, dia in enumerate(validDias)],
                            row_key='MC Index'
                        )

    def on_reset():
        # Clear image and result containers
        image_container.clear()
        result_container.clear()
        
        image_container.set_source('')

        # Reset inputs to default values
        mag.value = 0.58
        min_radius_input.value = 30
        max_radius_input.value = 70
        dist_input.value = 70
        sens_input.value = 33
        
        ui.notify("All values have been reset to default values.", type="positive")


    with ui.row().classes('w-full items-center').style('margin-bottom: 20px;'):
        ui.button('Back', on_click=ui.navigate.back).style('margin-left: 10px')
        ui.label('Circular Particle Counter').style('font-size: 280%; text-align: center; display: block; margin-left: auto; margin-right: auto;')

    with ui.row().classes('w-full no-wrap flex-start').style('justify-content: space-evenly;'):
        # Left Pane: Details
        with ui.card().classes('items-center').style('width: 25%'):
            def handle_upload(e: events.UploadEventArguments) -> None:
                global uploaded_image_content
                uploaded_image_content = e.content.read()

            ui.label('Upload image below: ')
            ui.upload(on_upload=handle_upload, multiple=False).style('width: 80%;').props('auto-upload')

            mag = ui.input('Scale (um per 1 px):', value=0.58)
            mag.props("size=40")

            min_radius_input = ui.input('Min radius (um):', value=30)
            min_radius_input.props("size=40")

            max_radius_input = ui.input('Max radius (um):', value=70)
            max_radius_input.props("size=40")

            dist_input = ui.input('Min distance(um) between centres:', value=70)
            dist_input.props("size=40")

            ui.label('Sensitivity level: ')
            sens_input = ui.slider(min=0, max=100, step=1, value=33).props('label-always').style('padding: 20px')

            with ui.row():
                ui.button('Submit', on_click=on_submit)
                ui.button('Reset', on_click=on_reset).style('margin-left: 10px')

        # Right Pane: Image Display
        with ui.column().classes('no-wrap items-center').style('width: 60%;'):
            image_container = ui.image().style('width: 600px')  # Image placeholder

            # Placeholder for the analysis results card
            result_container = ui.column().style('width: 100%')

@ui.page('/irregularCount')
def irregularMC():
    global uploaded_image_content, model, device, results

    device = "cpu"
    model = FastSAM('./weights/FastSAM-s.pt')

    with ui.header(elevated=True).style('background-color: white'):
        ui.image('./IMGS/SmartMCsLogo.png').style('width: 45px')

    # Define the on_submit function
    async def on_submit():
        if uploaded_image_content is None:
            ui.notify("Please upload an image before submitting.", type="warning")
            return

        if results is None or results[0].masks is None or len(results[0].masks.data) == 0:
            ui.notify("No particles detected. Upload another photo.", type="warning")
            return
        
        # Clear previous results
        image_container.clear()
        result_container.clear()

        factor = float(mag.value)
        if factor <= 0: 
            ui.notify("Error: Invalid scale factor", type="warning")
            return


        # Filter and process masks
        filtered_results = []
        mask_areas = []

        masks = results[0].masks
        filtered_masks = filter_masks_by_size(masks)
        non_overlapping_masks = filter_overlapping_masks(filtered_masks)
        results[0].masks.data = torch.stack(non_overlapping_masks)
        mask_areas = [calculate_mask_area(mask) for mask in non_overlapping_masks]
        filtered_results.append(results[0])

        prompt_process = FastSAMPrompt(input_image, filtered_results, device=device)
        ann = prompt_process.everything_prompt()
        
        # Compress and save the processed image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            output_path = temp_file.name
            prompt_process.plot(
                annotations=ann,
                output_path=output_path,
                withContours=False,
                better_quality=False,
            )

        # Optimize image file by compressing it further
        optimized_image = cv2.imread(output_path)
        _, compressed_image = cv2.imencode('.jpg', optimized_image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Convert to Base64
        base64_image = base64.b64encode(compressed_image).decode('utf-8')


        # Update the image container with the processed image
        image_container.set_source(f'data:image/jpeg;base64,{base64_image}')

        mask_dias = []
        total_count = len(mask_areas)
        for area in mask_areas:
            dia = 2 * math.sqrt(area / math.pi) / factor
            mask_dias.append(dia)
        mask_dias.sort()

        def open_calculate():
            new = {'dias': mask_dias} 
            app.storage.user.update(data=new) 
            ui.navigate.to('calculate')
        

        with result_container:
            with ui.card().classes('items-center').style('width: 100%'):
                ui.label('Analysis Results').style('font-size: 125%')
                with ui.row().style('width: 100%'): 
                    with ui.column().classes('items-center').style('width: 60%'):
                        with ui.row().style('width: 100%').classes('items-center'):
                            with ui.column().classes('items-center').style('width: 45%'):
                                ui.label('Particle Count: ')
                                ui.label(total_count).style('font-size: 300%')
                            with ui.column().classes('items-center').style('width: 45%'):
                                ui.label('Avg. Diameter (um): ')
                                mean = 0 if len(mask_dias) == 0 else statistics.mean(mask_dias)
                                ui.label(f"{mean:.3f}").style('font-size: 300%')
                            with ui.column().classes('items-center').style('width: 45%'):
                                ui.label('Stdev (um): ')
                                std = 0 if len(mask_dias) == 0 else (statistics.pstdev(mask_dias))
                                ui.label(f"{std:.3f}").style('font-size: 300%')
                                ui.space()

                            with ui.column().classes('items-center').style('width: 45%'):
                                ui.label('Coeff. of variation: ')
                                coeff = 0 if len(mask_dias) == 0 else (statistics.pstdev(mask_dias)/mean)
                                ui.label(f"{coeff:.3f}").style('font-size: 300%')
                                ui.space()
                        with ui.row().style('width: 100%, margin: 10px').classes('items-center'):
                            ui.label('Click here for further calculations and a graph representation.')
                        ui.button('calculate',  on_click=open_calculate)

                    with ui.column().classes('items-center').style('width: 30%'):
                        ui.label('List of Diameters(um): ')
                        async def copy_table_to_clipboard():
                            rows = diaList.rows
                            clipboard_data = "MC Index\tdia(um)\n"
                            ui.notify("Table successfully copied to clipboard.", type="positive")
                            for row in rows:
                                clipboard_data += f"{row['MC Index']}\t{row['dia(um)']}\n"
                            await ui.run_javascript(f'navigator.clipboard.writeText(`{clipboard_data}`)')

                        ui.button('Copy to Clipboard', on_click=copy_table_to_clipboard)
                        diaList = ui.table(
                            columns=[
                                {'name': 'MC Index', 'label': 'MC Index', 'field': 'MC Index', 'align': 'left'},
                                {'name': 'dia(um)', 'label': 'Diameter (um)', 'field': 'dia(um)', 'align': 'right'},
                            ],
                            rows=[{'MC Index': j + 1, 'dia(um)': f'{dia:.3f}'} for j, dia in enumerate(mask_dias)],
                            row_key='MC Index'
                        )

    with ui.row().classes('w-full items-center').style('margin-bottom: 20px;'):
        ui.button('Back', on_click=ui.navigate.back).style('margin-left: 10px')
        ui.label('Irregular Particle Counter').style('font-size: 280%; text-align: center; display: block; margin-left: auto; margin-right: auto;')

    with ui.row().classes('w-full no-wrap flex-start').style('justify-content: space-evenly;'):
    
        # Left Pane: Details
        with ui.card().classes('items-center').style('width: 25%'):
            def handle_upload(e: events.UploadEventArguments) -> None:
                global uploaded_image_content, results, input_image
                uploaded_image_content = e.content.read()

                def resize_image(image, max_width=1000):
                    original_width, original_height = image.size
                    
                    if original_width > max_width:
                        new_height = int((max_width / original_width) * original_height)
                        resized_img = image.resize((max_width, new_height), Image.LANCZOS)
                        return resized_img
                    else:
                        return image

                # Convert the uploaded file to an OpenCV image
                nparr = np.frombuffer(uploaded_image_content, np.uint8)
                original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if original_image is None:
                    ui.notify("Error: Could not read the image.")
                    return

                # Convert the uploaded image to a PIL image
                input_image = Image.open(io.BytesIO(uploaded_image_content))
                input_image = input_image.convert("RGB")

                # Resize the image if necessary
                input_image = resize_image(input_image, max_width=1000)
                
                # Pass the resized image to the model
                results = model(
                    input_image,
                    device=device,
                    retina_masks=True,
                    imgsz=1024,
                    conf=0.4,
                    iou=0.9    
                )

            ui.label('Upload image below: ')
            ui.upload(on_upload=handle_upload).style('width: 80%;').props('auto-upload')

            mag = ui.input('Scale (um per 1 px):', value=0.58)
            mag.props("size=40")

            def on_reset():
                # Clear image and result containers
                image_container.clear()
                result_container.clear()
                
                image_container.set_source('')

                # Reset inputs to default values
                mag.value = 0.58
                
                ui.notify("All values have been reset to default values.", type="positive")

            with ui.row():
                ui.button('Submit', on_click=on_submit)
                ui.button('Reset', on_click=on_reset).style('margin-left: 10px')

        # Right Pane: Image Display
        with ui.column().classes('no-wrap items-center').style('width: 60%;'):
            image_container = ui.image().style('width: 600px')  # Image placeholder
            result_container = ui.column().style('width: 100%')

def process_circular_image(image, minR, maxR, distLevel, sensLevel):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurLevel = 3 if maxR < 40 else 7
    blur = cv2.GaussianBlur(gray, (blurLevel, blurLevel), 2)

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=distLevel, param1=sensLevel, param2=22, minRadius=minR, maxRadius=maxR)
    
    if circles is not None:
        circles = circles[0]
        diameters = []
        output_image = image.copy()
        valid_circles = []

        for circle in circles:
            x = int(circle[0])
            y = int(circle[1])
            radius = int(circle[2])

            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), radius, 255, -1)
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            mask_pixels = masked_gray[mask == 255]
            avg_intensity = np.mean(mask_pixels)

            if (avg_intensity > 10) and (radius <= maxR):
                valid_circles.append(circle)
                diameter = radius * 2
                diameters.append(round(diameter, 3))
                
                # Draw the valid circle on the output image
                cv2.circle(output_image, (x, y), radius, (0, 255, 0), 2)
                cv2.circle(output_image, (x, y), 2, (0, 0, 255), 3)

        count = len(valid_circles)
        return count, diameters, output_image
    
    else:
        return 0, [], image

# Function to filter masks by size
def filter_masks_by_size(masks, max_size=250, min_size=25):
    filtered_masks = []
    for mask in masks.data:
        mask_np = mask.cpu().numpy()
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        if width <= max_size and height <= max_size and width > min_size and height > min_size:
            filtered_masks.append(mask)
    return filtered_masks

# Function to calculate IoU between two masks
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# Function to filter overlapping masks
def filter_overlapping_masks(masks, iou_threshold=0.05):
    keep_masks = []
    for i, mask1 in enumerate(masks):
        overlap = False
        mask1_np = mask1.cpu().numpy()
        for mask2 in keep_masks:
            mask2_np = mask2.cpu().numpy()
            iou = calculate_iou(mask1_np, mask2_np)
            if iou > iou_threshold:
                overlap = True
                break
        if not overlap:
            keep_masks.append(mask1)
    return keep_masks

# Function to calculate mask area
def calculate_mask_area(mask):
    mask_np = mask.cpu().numpy()
    return np.sum(mask_np)

ui.run(storage_secret='smartMCs', favicon="./IMGS/favicon.ico", title='Particle Counter')