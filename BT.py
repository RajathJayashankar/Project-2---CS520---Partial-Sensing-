import timeit

class BackTracking:

    def __init__(self, csp):

        self.csp = csp # CSP Objectis passed here 
        self.runtime = 0 # Exit condition for runtime


    def extractMRVNodeVariable(self):
        #Remove the node with minimum sized length constaint equation from list of Unassigned nodes
        md = -1
        mv = None
        for v in self.Unassigned_Nodes:
            if md < 0:
                md = v.cur_domain_size()
                mv = v
            elif v.cur_domain_size() < md:
                md = v.cur_domain_size()
                mv = v
        self.Unassigned_Nodes.remove(mv)
        return mv

    def restoreNodeVaraible(self, var):
        self.unasgn_vars.append(var) # Add Variable back to the list of unassigned constraint node variable 
        
    def Back_Track_Search(self,propagator):
        stime = time.process_time() # Start Timer
        
        self.Unassigned_Nodes = []
        for v in self.csp.vars:
            if not v.is_Inferred():
                self.Unassigned_Nodes.append(v)

        status = propagator(self.csp) #initial propagate no assigned variables.
        
        if status == False:
            print("CSP detected contradiction")
        else:
            status = self.Backtrack_Recurse(propagator, 1)   #now do recursive search


    def Backtrack_Recurse(self, propagator, level):

        # Return True if we find a solution, False if further seach is needed           
        if not self.Unassigned_Nodes:
            #all variables assigned
            return True
        else:
            var = self.extractMRVNodeVariable()
            for val in var():
                var.assign_a_value(val)
                status = propagator(self.csp, var) #Check is assignments are correct and valid
                if status: #If status return True, 
                    if self.Backtrack_Recurse(propagator, level+1):
                        return True
                var.make_Not_Inferred()
            self.restoreNodeVaraible(var)
            return False