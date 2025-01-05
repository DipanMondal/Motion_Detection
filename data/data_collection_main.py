from DataCollector import DataCollection

ob = DataCollection(1)
ob.run(label='standing ')
ob.append(path='data',file_name='skeleton_data.csv')
ob.release()
