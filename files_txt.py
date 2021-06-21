import numpy as np

def write_to_file(file_name_path = 'filename'):

    """
        writing variables / strings from List to txt file .
    """
    
    line1 = ['eldad', 63623612 , tuple(['ssdg' , 12642151.32])]
    line2 = ['asfafaeldad', 114124122 , tuple(['aaass' , 442261514.32])]
    line3 = 'rotem eldad orly amit omri'
             
    lines = [f'{line1}', f'{line2}',line3]  

    with open(f'{file_name_path}' + '.txt', 'w',encoding="utf8") as f:
        
       f.write('\n'.join(lines)) 


def write_to_file_using_print_function(file_name_path = 'filename'):

    with open(f'{file_name_path}' + '.txt', 'w',encoding="utf8") as f:
        
        print('using print function to write to a file', file = f )
        

def main(): 
    
    # write_to_file('file')
    # write_to_file_using_print_function('file_print')
    pass 

if __name__ == "__main__":

    main()