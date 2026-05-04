import { type ComponentType, type PropsWithChildren } from 'react';

import styles from '../Services.module.css';

interface Props {
  heading?: string;
  icon: ComponentType<{ className?: string }>;
}

function Service(props: PropsWithChildren<Props>) {
  const { heading, icon: Icon, children } = props;

  return (
    <div className={styles.service}>
      <Icon className={styles.icon} />
      <div className={styles.content}>
        {heading && <h2 className={styles.heading}>{heading}</h2>}
        {children}
      </div>
    </div>
  );
}

export default Service;
