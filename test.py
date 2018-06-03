
aa = ['', 'oz', 'Full', 'size', 'is', 'oz', 'for', 'rm', 'in', 'Sephora']
bb = [b for b in filter(lambda s:s!='', aa)]

print(' '.join(bb))
