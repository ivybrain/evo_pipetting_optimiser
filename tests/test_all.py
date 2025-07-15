'''
Execute by calling
pytest
in the command line from the root directory.
'''
import evo_pipetting_optimiser

# This is sample code. Feel free to delete any of it
def test_rect_area():
    '''Tests the rect area function'''
    assert evo_pipetting_optimiser.rect_area(3, 5) == 15
