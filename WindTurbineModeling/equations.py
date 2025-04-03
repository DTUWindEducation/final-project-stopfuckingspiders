def calc_flow_angle(axial_induction, tangential_induction, 
               inflow_wind_speed, rot_speed, span_position):
    
    numerator = (1 - axial_induction) * inflow_wind_speed
    denominator = (1 + tangential_induction) * rot_speed * span_position
    
    flow_angle = np.arctan(numerator / denominator)

    return flow_angle

def calc_local_angle_of_attack(flow_angle, blade_pitch_angle, r):
    
    twist_angle = calc_twist_angle(r)
    
    local_angle_of_attack = flow_angle  - (blade_pitch_angle + 
                                           twist_angle)
    local_angle_of_attack = 0
    return local_angle_of_attack

# Local solidity (sigma)
def calc_local_solidity(r, B):

    c = lambda r: calc_chord_length(r) 
    Nominator = c * B
    Denominator = 2*np.pi*r

    sigma = Nominator/Denominator

    return sigma
# Calculate the tip speed ratio (lambda)
def calc_speed_ratio(rot_speed, inflow_wind_speed, rotor_radius):
    # lambda = omega * R / V
    speed_ratio = rot_speed * rotor_radius / inflow_wind_speed
    
    return  speed_ratio

def calc_chord_length(r):
    return 1

def calc_twist_angle(r):
    return 1