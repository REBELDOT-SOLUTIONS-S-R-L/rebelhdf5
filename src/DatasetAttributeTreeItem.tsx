import { createContext, useContext } from 'react';
import { FiFile, FiFolder } from 'react-icons/fi';

import styles from './DatasetAttributesPage.module.css';
import {
  type AttributeTreeNode,
  prettifyLeafValue,
} from './DatasetAttributesPage.utils';

export interface ExpandedLeafContextValue {
  isExpanded: (path: string) => boolean;
  toggle: (path: string) => void;
}

export const ExpandedLeafContext = createContext<ExpandedLeafContextValue>({
  isExpanded: () => false,
  toggle: () => undefined,
});

export function DatasetAttributeTreeItem({
  node,
  depth,
}: {
  node: AttributeTreeNode;
  depth: number;
}) {
  const { isExpanded, toggle } = useContext(ExpandedLeafContext);

  if (node.children) {
    return (
      <div className={styles.treeBranch}>
        <div
          className={styles.treeBranchRow}
          style={{ paddingLeft: `${depth * 1.1}rem` }}
          title={node.path}
        >
          <FiFolder aria-hidden className={styles.treeBranchIcon} />
          <span className={styles.treeBranchName}>{node.name}</span>
          {node.value && (
            <small className={styles.treeMeta}>{node.value}</small>
          )}
        </div>
        {node.children.length > 0 && (
          <div className={styles.treeChildren}>
            {node.children.map((child) => (
              <DatasetAttributeTreeItem
                key={child.path}
                node={child}
                depth={depth + 1}
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  const expanded = isExpanded(node.path);
  const fullValue = expanded ? prettifyLeafValue(node.value) : null;

  return (
    <div className={styles.treeLeafWrapper}>
      <button
        type="button"
        className={`${styles.treeLeaf} ${expanded ? styles.treeLeafExpanded : ''}`}
        style={{ paddingLeft: `${depth * 1.1}rem` }}
        title={node.path}
        aria-expanded={expanded}
        onClick={() => {
          toggle(node.path);
        }}
      >
        <FiFile aria-hidden className={styles.treeLeafIcon} />
        <span className={styles.treeLeafLabel}>{node.name}</span>
        <small className={styles.treeMeta}>{node.value}</small>
      </button>
      {expanded && fullValue !== null && (
        <pre
          className={styles.treeLeafFull}
          style={{ paddingLeft: `${depth * 1.1 + 1.6}rem` }}
        >
          {fullValue}
        </pre>
      )}
    </div>
  );
}
