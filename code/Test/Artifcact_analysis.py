import os
import xml.etree.ElementTree as ET

def find_artifacts_in_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        print(f"Parse error in {xml_path}: {e}")
        return set()
    
    root = tree.getroot()
    artifacts = set()

    # Collect all Name elements that are under ScoredEventSettings
    excluded_names = set()
    for scored_event_settings in root.iter('ScoredEventSettings'):
        for name_elem in scored_event_settings.iter('Name'):
            excluded_names.add(name_elem)

    # Iterate over all Name elements and exclude those under ScoredEventSettings
    for name_elem in root.iter('Name'):
        if name_elem in excluded_names:
            continue
        if name_elem.text and 'artifact' in name_elem.text.lower():
            artifacts.add(name_elem.text.strip())
    
    return artifacts

def scan_artifacts_in_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Directory does not exist: {directory_path}")
        return
    
    all_artifacts = set()
    for filename in os.listdir(directory_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(directory_path, filename)
            artifacts = find_artifacts_in_xml(file_path)
            print(f'File: {filename}\n  Artifacts found: {artifacts}\n')
            all_artifacts.update(artifacts)
    
    print('Summary of all artifact types across files:')
    for artifact in sorted(all_artifacts):
        print(f'- {artifact}')

if __name__ == '__main__':
    directory = r'D:\cfs\polysomnography\annotations-events-profusion'
    scan_artifacts_in_directory(directory)
