import numpy as np

def read_and_process_from_file_line_by_line(file_name_path = 'filename'):
     
     s =  ''
     
     with open(f'{file_name_path}' + '.txt', 'r',encoding="utf8") as f:

         for line in f:

            s+= line

     print(s)

def read_from_file(file_name_path = 'filename'):

    with open(f'{file_name_path}' + '.txt', 'r',encoding="utf8") as f:
        
        all_lines = f.read() #read till the end of the file 
        print(all_lines)

        line = f.readline() #read the next line in the file iterable  
        print(line)

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
    
    # write_to_file(file_name_path = 'filename')
    # write_to_file_using_print_function(file = 'file_print')
    # read_from_file()
    read_and_process_from_file_line_by_line()
    pass 

if __name__ == "__main__":

    main()