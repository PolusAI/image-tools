import click


@click.command()
@click.option("--name", "-r", default="world", type=str)
def main(name):
    """Parse a timezone and greet a location a number of times."""
    print(f"Hello {name}!")


if __name__ == "__main__":
    main()
