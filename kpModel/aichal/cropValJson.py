#!/usr/bin/env python
import json

def main():
    with open('valBak.json', 'r') as f:
        data = json.load(f)
    data = data[:1000]

    with open('val.json', 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    main()
