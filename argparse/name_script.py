import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-n","--name",required=True,help="name of the user")
args=vars(ap.parse_args())

print("Hi ther e{}, it's nice to meet you!".format(args["name"]))

print(args)
