import time
from multiprocessing import Process, Queue, Array, Value, cpu_count
import sys

class parallelization():
    """
    Parallelization Class for Function Calls.
    """
    def __init__(self, maximum_number_of_cores=cpu_count()):
        """
        Initialize a Parallelization Class.
        
        Parameters:
        ----------
        (optional) maximum_number_of_cores: (int) Maximal number of available CPUs, automatic assignment is
                                            all available ones.
        """
        self.maximum_number_of_cores = maximum_number_of_cores
        return None
        
    def worker(self, q, process_number, cpus, function, number_of_voxels, *args):
        done = False 
        while done != True:
            
            with self.calculations.get_lock():
                vox = sum(self.calculations)
                self.calculations[process_number] += 1
        
            # Everything done
            if vox >= number_of_voxels:
                done = True
                break
        
            # Calculate approximate time remaining for fitting
            percentage_realistic = int(vox*1.0/number_of_voxels*100.0)
            
            if percentage_realistic > self.percent.value:
                self.percent.value = percentage_realistic

                # Three-point estimation (PERT-Estimation)
                percentage_pessimistic = int(min(self.calculations)*self.number_of_cores*1.0/number_of_voxels*100.0)
                percentage_optimistic = int(max(self.calculations)*self.number_of_cores*1.0/number_of_voxels*100.0)
                
                if percentage_pessimistic == 0:
                    percentage_pessimistic = 1
                
                t_delta = time.time() - self.starting_time         
                time_remaining_optimistic = t_delta/percentage_optimistic*(100.0-percentage_optimistic)/60.0
                time_remaining_realistic = t_delta/percentage_realistic*(100.0-percentage_realistic)/60.0
                time_remaining_pessimistic = t_delta/percentage_pessimistic*(100.0-percentage_pessimistic)/60.0
                time_remaining = (time_remaining_optimistic + 4.0*time_remaining_realistic + time_remaining_pessimistic)/6.0 
    
                sys.stdout.write('{:3.0f}%  {:8.2f}min remaining\n'.format(percentage_realistic, time_remaining))
    
            params = ()
            for param in args[0]:
                if len(param) != 1:
                    params = params + (param[vox],)
                else:
                    params = params + (param[0],)
                    
            if len(function) != 1:
                result = function[vox](*params)
            else:
                result = function[0](*params)
    
            results = []
            results.append(vox)
            results.append(result)
            q.put(results)
        
    def start(self, function, number_of_voxels, *args):
        """"
        Starts the Parallelization of the Parallelization Class.
        
        Parameters:
        ----------
        function:           (List) List of functions that should be executed, if always the same 
                            function should be used, then a list containing only one function [function] has to be assigned.
        number_of_voxels:   (int) Amount of Processes to be parallelized.
        (optional) *args:   (Tuple of Lists) Parameter for executed function. ALL PARAMETER MUST BE ASSIGNED, even optional
                            ones if one optional parameter should be used, in the correct order, without name assignement.
                            If always the same parameter should be used, then a list containing one parameter [parameter]
                            has to be assigned.
                            
        Returns:
        --------
        List of return values from the excecuted function. The length equals number_of_voxels.
                            
        Example:
        -------
        models = p.start([TensorModel], 4, [gtab1, gtab2, gtab3, gtab4])
        fits = p.start([i.fit for i in models], 4, [data1, data2, data3, data4], [TE])
        """
        #Multicore Calculation
        self.number_of_cores = cpu_count()
        if self.maximum_number_of_cores < self.number_of_cores:
            self.number_of_cores = self.maximum_number_of_cores
        
        self.calculations = Array('i', self.number_of_cores)
        self.percent = Value('i', 0)

        self.percent.value = 0
        self.starting_time = time.time() 
        q = Queue()
        processes = []

        print 'Parallelization starts on', self.number_of_cores, 'CPUs.'
        
        for i in range(self.number_of_cores):
            self.calculations[i] = 0
            processes.append(Process(target=self.worker, args=(q, i, self.number_of_cores, function, number_of_voxels, args)))
            processes[-1].start()  
        
        return_values = [None] * number_of_voxels
        
        for i in range(number_of_voxels):
            results = q.get()
            vox = results[0]
            return_values[vox] = results[1]
        
        # Exit the completed processes
        for i in range(self.number_of_cores):
            processes[i].join()

        sys.stdout.write('{:3.0f}%  {:8.2f}min remaining\n'.format(100.0, 0.0))
        sys.stdout.write('Total Time needed: {:4.2f}min\n'.format(((time.time() - self.starting_time)/60.0)))
    
        return return_values