"""
EMProt: automated modeling of proteins from cryo-EM maps
"""
def main():
    import argparse
    import warnings
    import emprot

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"EMProt {emprot.__version__}",
    )

    # Suppress some warnings
    warnings.filterwarnings("ignore")

    import emprot.run
    import emprot.eval
    import emprot.clean
    import emprot.fit
    import emprot.refine

    modules = {
        "build": emprot.run,
        "eval": emprot.eval, 
        "clean": emprot.clean,
        "fit": emprot.fit,
        "refine": emprot.refine, 
    }

    subparsers = parser.add_subparsers(title="Modules",)
    subparsers.required = "True"

    for key in modules:
        module_parser = subparsers.add_parser(
            key,
            description=modules[key].__doc__,
            formatter_class=argparse.RawTextHelpFormatter,
        )
        modules[key].add_args(module_parser)
        module_parser.set_defaults(func=modules[key].main)

    try:
        args = parser.parse_args()
        args.func(args)
    except TypeError:
        parser.print_help()

if __name__ == '__main__':
    main()
