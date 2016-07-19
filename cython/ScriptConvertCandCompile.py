
import os



print('Create c file')


for el in os.listdir(os.getcwd()):
	if '.py' in el:
		elWithoutExt = el.split('.')[0]
		commandC = 'cython ' + el + ' -o VersionC/' + elWithoutExt + '.c --embed=WinMain'
		os.system(commandC)
		
		
print('End of generation')