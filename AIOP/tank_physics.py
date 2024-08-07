'''function that calculates a missing flow from a drum'''
import pandas as pd

#initiate a physics tag for a tank level if you want.
# input_tags = [5,6,7]
# output_tags = [4]
# span_in = 96
# diameter_ft = 6
# orientation = 'vertical'
# flow_units = 'gpm'
# length_ft = 10
# sim.initiate_physics(pvIndex=PV_INDEX,input_tags=input_tags,output_tags=output_tags,
#                      span_in=span_in,diameter_ft=diameter_ft,orientation=orientation,
#                      flow_units = flow_units,length_ft = length_ft)


# if sim.physics:
#     config['physics_config'] = {
#         'input_tags':input_tags,
#         'output_tags':output_tags,
#         'span_in':span_in,
#         'diameter_ft':diameter_ft,
#         'length_ft':length_ft,
#         'orientation':orientation,
#         'flow_units':flow_units
#         }
    

def pv_diff(data,pv,span_in,diameter_ft,orientation,flow_units,
                      scanrate,length_ft = 0):
    '''
    function to create a new column in a pandas dataframe with
    a the flow rate of accumulated liquid in a vessel per scan.
    units  = 'bpd','gpm'
    orientation = 'vertical', 'horozontal'
    scanrate in seconds
    returns volume in barrels or gallons
    '''
    if flow_units == 'bpd':
        volume_conversion = 0.178 #bbl_per_ft3
        rate_conversion = (60/scanrate)*60*24
    elif flow_units == 'gpm':
        volume_conversion = 7.48 #gallons_per_ft3
        rate_conversion = 60/scanrate
    
    if orientation == 'vertical':
        area = 3.14159*(diameter_ft/2)**2 #ft3
    elif orientation =='horozontal':
        area = diameter_ft*length_ft #estimate...

    #percent change per scan
    dldt = data[1:][pv].reset_index(drop=True) \
    - data[:-1][pv].reset_index(drop=True)
    dldt = pd.concat([pd.Series(dldt),pd.Series([0])],ignore_index=True)
    
    dvdt = (dldt*(span_in/100)/12)*area*volume_conversion #(barrels,gallons) / scan

    acc_rate = dvdt*rate_conversion

    return acc_rate
