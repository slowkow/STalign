'''Copy what the explorer page needs from build/ into docs/_extra/explorer/assets/.

The page (docs/_extra/explorer/index.html) shows the PNGs directly and reads its numbers from
small JavaScript files in assets/data/, one per dataset, each of which only assigns a JSON value:

    (window.STALIGN_EXPLORER = window.STALIGN_EXPLORER || {})["manifest"] = {...};

Scripts load from file:// URLs, where fetch() of a JSON file is blocked, so the page also works
when index.html is opened directly from disk.  Run after the asset scripts (run_all.sh does).
'''
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, 'build')
SITE = os.path.join(os.path.dirname(HERE), '_extra', 'explorer')
OUT = os.path.join(SITE, 'assets')

# PNGs the page does not show
SKIP = {'rasterize/cells_thumbnail.png', 'mixture/target_full.png', 'mixture/target_cropped.png'}


def read(rel):
    with open(os.path.join(BUILD, rel)) as f:
        return json.load(f)


def write_js(name, value):
    path = os.path.join(OUT, 'data', name + '.js')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'(window.STALIGN_EXPLORER = window.STALIGN_EXPLORER || {{}})[{json.dumps(name)}] = {text};\n')
    return os.path.getsize(path)


def main():
    if not os.path.isfile(os.path.join(BUILD, 'manifest.json')):
        sys.exit(f'{BUILD}/manifest.json not found: run run_all.sh first')
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    n, size = 0, 0
    for src in sorted(glob.glob(os.path.join(BUILD, '**', '*.png'), recursive=True)):
        rel = os.path.relpath(src, BUILD).replace(os.sep, '/')
        if rel in SKIP:
            continue
        dst = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        n, size = n + 1, size + os.path.getsize(dst)
    manifest = read('manifest.json')
    size += write_js('manifest', manifest)
    size += write_js('zoom_cells', read(manifest['rasterize']['zoom_cells']))
    for run in manifest['regularization']['runs']:
        size += write_js(run['tag'], dict(grid=read(run['grid']), energy=read(run['energy'])))
    n += 2 + len(manifest['regularization']['runs'])
    print(f'packaged {n} files, {size / 1e6:.1f} MB, into {os.path.relpath(OUT, os.getcwd())}')


if __name__ == '__main__':
    main()
