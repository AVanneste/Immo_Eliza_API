import json


# Opening JSON files
# with open('input_data.json') as json_file1:
#     input_data = json.load(json_file1)
    
with open('correlations.json') as json_file2:
    correlations = json.load(json_file2)

with open('medians.json') as json_file3:
    medians = json.load(json_file3)

with open('pc_coefs.json') as json_file4:
    pc_coefs = json.load(json_file4)



def preprocess(data):

    print(type(data))
    
    data = data.data

    # print('this is data: ',data)

    predict = []

    property_type = data.property_type


    post_code = data.zip_code
    if 1000<=post_code<10000:
        post_code = pc_coefs[str(post_code)]
        predict.append(post_code)
    else:
        print('Please input a valid post code')


    garden = data.garden
    if garden:
        garden = correlations[property_type]['Garden']
        predict.append(garden)
    else:
        predict.append(0)


    terrace = data.terrace
    if terrace:
        terrace = correlations[property_type]['Terrace']
        predict.append(terrace)
    else:
        predict.append(0)


    area = data.area
    if area>0:
        area = (area/medians[property_type]['Living area'])*correlations[property_type]['Living area']
        predict.append(area)
    else:
        print('Please input an integer greater than 0')


    bedrooms = data.rooms_number
    if type(bedrooms) == int and bedrooms>0:
        bedrooms = (bedrooms/medians[property_type]['Bedrooms'])*correlations[property_type]['Bedrooms']
        predict.append(bedrooms)
    else:
        print('Please input an integer greater than 0')


    plot_surface = data.land_area
    if property_type == 'APARTMENT':
        predict.append(0)
    elif type(plot_surface) == int and plot_surface>0:
        plot_surface = (plot_surface/medians[property_type]['Surface of the plot'])*correlations[property_type]['Surface of the plot']
        predict.append(plot_surface)
    else:
        plot_surface = correlations[property_type]['Surface of the plot']
        predict.append(plot_surface)


    garden_area = data.garden_area
    if type(garden_area) == int and garden_area>0:
        garden_area = (garden_area/medians[property_type]['Garden surface'])*correlations[property_type]['Garden surface']
        predict.append(garden_area)
    else:
        predict.append(0)


    swimming_pool = data.swimming_pool
    if swimming_pool:
        swimming_pool = correlations[property_type]['Swimming pool']
        predict.append(swimming_pool)
    else:
        predict.append(0)


    open_fire = data.open_fire
    if open_fire:
        open_fire = correlations[property_type]['Open fire']
        predict.append(open_fire)
    else:
        predict.append(0)


    terrace_area = data.terrace_area
    if type(terrace_area) == int and terrace_area>0:
        terrace_area = (terrace_area/medians[property_type]['Terrace surface'])*correlations[property_type]['Terrace surface']
        predict.append(terrace_area)
    else:
        predict.append(0)


    facades = data.facades_number
    if type(facades) == int and facades>0:
        facades = (facades/medians[property_type]['Number of frontages'])*correlations[property_type]['Number of frontages']
        predict.append(facades)
    else:
        facades = correlations[property_type]['Number of frontages']
        predict.append(facades)


    build_condition = data.building_state
    if build_condition == 'NEW' or build_condition == 'JUST RENOVATED':
        build_condition = (3/medians[property_type]['Building condition'])*correlations[property_type]['Building condition']
        predict.append(build_condition)
    elif build_condition == 'GOOD' or build_condition == None:
        build_condition = (2/medians[property_type]['Building condition'])*correlations[property_type]['Building condition']
        predict.append(build_condition)
    elif build_condition == 'TO RENOVATE' or build_condition == 'TO REBUILD':
        build_condition = (1/medians[property_type]['Building condition'])*correlations[property_type]['Building condition']
        predict.append(build_condition)



    kitchen = data.equipped_kitchen
    if kitchen:
        kitchen = correlations[property_type]['Kitchen type']
        predict.append(kitchen)
    else:
        predict.append(0)


    # furnished = data['Furnished']
    # if furnished:
    #     furnished = correlations[property_type]['Furnished']
    #     predict.append(furnished)
    # else:
    #     predict.append(0)


    return predict




# x = preprocess(input_data)

# print(x)


