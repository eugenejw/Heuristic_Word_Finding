import inspect

class key_memoized(object):
    def __init__(self, func):
       self.func = func
       self.cache = {}

    def __call__(self, *args, **kwargs):
        
        key = self.key(args, kwargs)
        print "key is: {}".format(key)
        for each in kwargs.items():
            if isinstance(each[1], list):
                if len(each[1]) > 1:
                    print "QUIT*******************"
                    return self.func(*args, **kwargs)
        if key not in self.cache:
            self.cache[key] = self.func(*args, **kwargs)
            return self.cache[key]
        else:
            print "Result from Momery*********"
            return self.cache[key]

    def normalize_args(self, args, kwargs):
        spec = inspect.getargs(self.func.__code__).args
#        print "kwargs.items(): {}".format(kwargs.items())
        new_kwargs = []
        for each in kwargs.items():
            tmp_lst = []
#            print "working on each {0}, of type {1}".format(each, type(each))
            for e in each:
                if isinstance(e, list):
                    new_e = tuple(e)
                    tmp_lst.append(new_e)
#                    print "appended {}".format(new_e)
                    continue
                tmp_lst.append(e)
#                print "appended {}".format(e)
            new_kwargs.append(tuple(tmp_lst))
        
#        print "After tupling, kwargs.items(): {}".format(new_kwargs)

        return dict(new_kwargs + zip(spec, args))

    def key(self, args, kwargs):
        
        a = self.normalize_args(args, kwargs)
        return tuple(sorted(a.items()))

@key_memoized
def foo(bar, baz, spam, spam2=0):
    print 'Real Code: calling foo: bar=%r baz=%r spam=%r' % (bar, baz, spam)
    return bar + baz + spam[0] + spam[0]

#print foo(1, 2, (3,))
#print foo(1, 2, (4,))
#print foo(1, 2, spam=(3,), spam2=[4]) 
#print foo(1, 2, spam=(3,), spam2=[4]) 

print foo(1, 2, spam=[3], spam2=[4]) 
print foo(1, 2, spam=[3], spam2=[4]) 
#print foo(1, 2, spam=[4,5], spam2=(4,5)) 
