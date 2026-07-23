import {
  FiChevronDown,
  FiChevronRight,
  FiFile,
  FiFolder,
} from 'react-icons/fi';
import { HiFolder } from 'react-icons/hi';

import styles from './DatasetProcessingPage.module.css';
import { type KeyTreeNode } from './DatasetProcessingPage.utils';

interface DatasetProcessingKeyTreeNodeProps {
  node: KeyTreeNode;
  depth: number;
  selectedKeys: ReadonlySet<string>;
  collapsedGroups: ReadonlySet<string>;
  onToggleKey: (keyPath: string) => void;
  onToggleGroup: (keyPaths: readonly string[]) => void;
  onToggleCollapse: (groupPath: string) => void;
}

export function DatasetProcessingKeyTreeNode({
  node,
  depth,
  selectedKeys,
  collapsedGroups,
  onToggleKey,
  onToggleGroup,
  onToggleCollapse,
}: DatasetProcessingKeyTreeNodeProps) {
  if (node.keyInfo) {
    return (
      <label
        className={styles.treeLeaf}
        style={{ paddingLeft: `${depth * 1.1 + 0.75}rem` }}
        title={node.fullPath}
      >
        <input
          type="checkbox"
          aria-label={`Select ${node.fullPath}`}
          checked={selectedKeys.has(node.fullPath)}
          onChange={() => {
            onToggleKey(node.fullPath);
          }}
        />
        <FiFile aria-hidden className={styles.treeLeafIcon} />
        <span className={styles.treeLeafLabel}>{node.name}</span>
        <small className={styles.treeMeta}>
          {node.keyInfo.availableInDemoCount}
        </small>
      </label>
    );
  }

  const selectedCount = node.leafKeyPaths.filter((keyPath) =>
    selectedKeys.has(keyPath),
  ).length;
  const allSelected = selectedCount === node.leafKeyPaths.length;
  const isCollapsed = collapsedGroups.has(node.fullPath);
  const FolderIcon = allSelected ? HiFolder : FiFolder;

  return (
    <div className={styles.treeBranch}>
      <div
        className={styles.treeBranchRow}
        style={{ paddingLeft: `${depth * 1.1}rem` }}
      >
        <button
          type="button"
          className={styles.treeBranchSelectBtn}
          title={`Toggle all keys under ${node.fullPath}`}
          aria-pressed={allSelected}
          onClick={() => {
            onToggleGroup(node.leafKeyPaths);
          }}
        >
          <FolderIcon aria-hidden className={styles.treeBranchIcon} />
        </button>
        <button
          type="button"
          className={styles.treeBranchLabel}
          title={node.fullPath}
          aria-expanded={!isCollapsed}
          onClick={() => {
            onToggleCollapse(node.fullPath);
          }}
        >
          {isCollapsed ? (
            <FiChevronRight aria-hidden className={styles.treeCollapseIcon} />
          ) : (
            <FiChevronDown aria-hidden className={styles.treeCollapseIcon} />
          )}
          <span className={styles.treeBranchName}>{node.name}</span>
        </button>
        <small className={styles.treeMeta}>
          {selectedCount}/{node.leafKeyPaths.length}
        </small>
      </div>
      {!isCollapsed && (
        <div className={styles.treeChildren}>
          {node.children.map((child) => (
            <DatasetProcessingKeyTreeNode
              key={child.fullPath}
              node={child}
              depth={depth + 1}
              selectedKeys={selectedKeys}
              collapsedGroups={collapsedGroups}
              onToggleKey={onToggleKey}
              onToggleGroup={onToggleGroup}
              onToggleCollapse={onToggleCollapse}
            />
          ))}
        </div>
      )}
    </div>
  );
}
