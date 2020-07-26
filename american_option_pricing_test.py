


from american_option_pricing import american_option


option_type = 'c'
fs = 98.95
x= 104
t = 9/252
r=0.103/100
b=0
v=0.2793

ans = american_option(option_type, fs, x, t, r, v)
#actual = 5.75
#print(actual)
print((0.62+0.35)/2)
print(ans[0])
