import BT

Confirmed_type = {"notConfirmed" : 1,"ConfirmedEmpty" : 2,"ConfirmedBlocked" : 3}

class CSP():
    #Initialize grid knowledge and Constraint solver
    def __init__(self, grid):
        self.grid = grid
        self.all_constraint_equations = []
        self.empty_node_variables = []
        self.blocked_node_variables = []

    

    # Method called during Path Traversal for Agent 4
    def Constraint_Solver(self):
        self.resolve_subsets()
        val = BT(self)
        return val

    #Creates Constraint Equations
    def create_constraint_equation_for_variable(self, node):
        row, col = node.get_pos() #Get row and col of Node
        for i in [-1, 0, 1]: # Iterate for 8 cells around you
            for j in [-1, 0, 1]:
                if (i == 0 and j == 0):
                    continue

                if (row + i >= 0 and col + j >= 0 and row + i < self.grid.rows and col + j < self.grid.rows):

                    # If a neighbour is Confirmed Empty, then do not add it to the constraint equation.
                    if self.grid[row + i, col + j].confirmed == Confirmed_type["ConfirmedEmpty"]:
                        continue

                    # If a neighbour is already confirmed Blocked, then do not add it to the equation but subtract the constraint value of the current variable -- Cx.
                    if self.grid[row + i, col + j].confirmed == Confirmed_type["ConfirmedBlocked"]:
                        node.Cx -= 1
                        continue

                    neighbour = self.grid[row + i, col + j]
                    node.add_constraint_variable(variable = neighbour)

        # Append the equation in the global equation list
        self.all_constraint_equations.append([node.constraint_equation, node.Cx])

    def remove_duplicates(self, array):

        # Create an empty list to store unique elements
        uniqueList = []

        # Iterate over the original list and for each element
        # add it to uniqueList, if its not already there.
        for element in array:
            if element not in uniqueList:
                uniqueList.append(element)

        # Return the list of unique elements
        return uniqueList

    def resolve_subsets(self):
        # Sort all equations in increasing order of their length
        self.all_constraint_equations = sorted(self.all_constraint_equations, key = lambda x: len(x[0]))

        # Start resolving subsets
        for equation in self.all_constraint_equations:
            for equation2 in self.all_constraint_equations:

                if equation == equation2 or not equation[0] or not equation2[0] or not equation[1] or not equation2[1]:
                    continue

                # Check if the equation is a subset of the other equations
                if set(equation[0]).issubset(set(equation2[0])):
                    equation2[0] = list(set(equation2[0]) - set(equation[0]))
                    equation2[1] -= equation[1]
                    continue

                # Check if the equation is a superset of the other equations
                if set(equation2[0]).issuperset(set(equation[0])):
                    equation[0] = list(set(equation[0]) - set(equation2[0]))
                    equation[1] -= equation2[1]

        # After resolving subsets, check if now we can get blocked and unbloced nodes
        self.Check_Constraint_Equations_BlockedAndFree() 

    def Check_Constraint_Equations_BlockedAndFree(self):

        for equation in self.all_constraint_equations.copy():

            # If the equation is empty i.e. all its contraint nodes are removed
            if len(equation) == 0 or len(equation[0]) == 0:
                self.all_constraint_equations.remove(equation)
                continue

            # If value is 0, all nodes in that equation are free nodes.
            if equation[1] == 0:
                self.all_constraint_equations.remove(equation)
                for free_nodes in equation[0]:
                    if not self.is_inferred():
                        self.make_inferred()
                        self.confirmed = Confirmed_type["ConfirmedEmpty"]
                continue

            # If value is equal to the length of the equation, then all the nodes in the equation are blocked nodes.
            if len(equation[0]) == equation[1]:
                self.all_constraint_equations.remove(equation)
                for BlockedNode in equation[0]:
                    if not self.env.flags[BlockedNode.row, BlockedNode.column] and BlockedNode not in self.mine_nodes:
                        self.BlockedNodes.append(BlockedNode)



