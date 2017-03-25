import sys

'''
Args allow to parse arguments from command line
How to use it:
1) Initialize Args class
==> my_args = Args(var1=(var_type, true_or_false), var2=(...), ...)
==> # true_or_false is a boolean used to inform if the argument is necessary or not
2) Transform arguments from the command line into class attributes of Args
==> my_args.resolve_args()
3) Access to your variables
==> my_args.var1
==> my_args.var2

'''

class Args:
  def __init__(self, **args):
    '''
    Must received variable_name=(value_type, boolean) couple
    where the boolean indicates if the argument is necesarry or optional (False)

    exemple: Args(name=(str, True), count=(int, False))

    '''
    self.args = args

  def resolve_args(self):
    '''
    Converts into class variable the arguments parsed from the command line

    all_vars: list of arguments passed

    '''
    tmp = dict(a.split("=") for a in sys.argv[1:])
    self.all_vars = tmp.keys()
    self.check(tmp.keys())
    for k, v in tmp.items():
      if k in self.args.keys():
        setattr(self, k, self.args[k][0](v))
      else:
        setattr(self, k, v)
  
  def check(self, received_args):
    '''
    Checks if necessary variables are missing

    '''
    for k, v in self.args.items():
      if v[1]:
        assert k in received_args, "ERROR: argument <{}> is missing".format(k)

#a = Args(name=(str,True), count=(int,False))
#a.resolve_args()
#print("name: {}".format(a.name))
#if hasattr(a, 'count'):
#  print("count: ", a.count)