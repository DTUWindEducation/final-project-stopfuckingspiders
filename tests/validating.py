import WindTurbineModeling.read_data as WTMread

from typing import Union
from pathlib import Path

def test_read_blade_definition(file_path: Union[str, Path]):
    """Check"""
    BlSpn_min_exp = 0.000000000000000
    BlSpn_max_exp = 116.9999315223028
    
    df = WTMread.read_blade_definition(file_path)
    BlSpn_min = df['BlSpn'].min()
    BlSpn_max = df['BlSpn'].max()
    
    assert BlSpn_min == BlSpn_min_exp
    assert BlSpn_max == BlSpn_max_exp