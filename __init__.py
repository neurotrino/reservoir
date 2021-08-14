import pkgutil
import sys


def load_all_modules_from_dir(dirname):
    for importer, package_name, _ in pkgutil.iter_modules([dirname]):
        full_package_name = '%s.%s' % (dirname, package_name)
        if full_package_name not in sys.modules:
            module = importer.find_module(package_name
                        ).load_module(full_package_name)
            print()
            print(module)
            print()

print()
print()
print()
print()
load_all_modules_from_dir('Foo')
print()
print()
print()
print()
