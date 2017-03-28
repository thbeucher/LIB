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
    Must received variable_name=(value_type, boolean, [default_value]) couple
    where the boolean indicates if the argument is necesarry or optional (False)
    default_value is optional argument

    exemple: Args(name=(str, True), count=(int, False, 0))

    '''
    self.args = args
    for k, v in args.items():
        if len(v) == 3:
            setattr(self, k, v[2])

  def resolve_args(self):
    '''
    Converts into class variable the arguments parsed from the command line

    all_vars: list of arguments passed

    '''
    # get all arguments in a dictionary
    tmp = dict(a.split("=") for a in sys.argv[1:])
    self.all_vars = tmp.keys()
    # check arguments
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

#a = Args(name=(str,True), count=(int,False, 0))
#a.resolve_args()
#print("name: {}".format(a.name))
#if hasattr(a, 'count'):
#  print("count: ", a.count)


'''
tt = "coucou"
def tt2():
  print('tt2')

class tt3:
  def __init__(self):
    print("tt3")

  def test(self):
    print("test OK")

mm = {'tt2':tt2, 'tt3': tt3}

myargs = Args(tt=(str, False, tt),tt2=(str, True),tt3=(str, True))
myargs.resolve_args()

# tt
myargs.tt

# tt2
mm[myargs.tt2]()

# tt3
a = mm[myargs.tt3]()
a.test()
'''
