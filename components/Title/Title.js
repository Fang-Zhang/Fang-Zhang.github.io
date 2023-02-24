import React from 'react';
import PropTypes from 'prop-types';
import Typography from '@mui/material/Typography';
import useStyles from './title-style';

export default function Title(props) {
  const { classes, cx } = useStyles();
  const { children, dark } = props;

  return (
    <div className={cx(classes.title, dark && classes.dark)}>
      <Typography variant="h4">
        {children}
      </Typography>
    </div>
  );
}

Title.propTypes = {
  children: PropTypes.node.isRequired,
  dark: PropTypes.bool,
};

Title.defaultProps = {
  dark: false,
};
